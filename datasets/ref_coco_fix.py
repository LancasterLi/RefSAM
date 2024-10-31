# 使用Refcoco/Refcoco+/Refcocog
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data
import torchvision

import os
from tqdm import tqdm
import argparse
import warnings
from per_segment_anything import sam_model_registry, SamPredictor
from transformers import T5EncoderModel, T5TokenizerFast
import logging
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import util.misc as utils
import datasets.samplers as samplers
import torch.distributed as dist
from einops import rearrange
from datasets.coco_eval import CocoEvaluator
from pycocotools import mask as coco_mask
from collections import namedtuple
import matplotlib.pyplot as plt
from PIL import Image

import opts
import refersam
import loss

import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
warnings.filterwarnings('ignore')


class ModulatedDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, image_set, return_masks, args):
        super(ModulatedDetection, self).__init__(img_folder, ann_file)
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.image_set = image_set
        self.args = args

    def __getitem__(self, idx):
        instance_check = False
        frame_num = self.args.frame_num ##
        while not instance_check:
            img, target = super(ModulatedDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            coco_img = self.coco.loadImgs(image_id)[0]
            caption = coco_img["caption"]
            dataset_name = coco_img["dataset_name"] if "dataset_name" in coco_img else None
            target = {"image_id": image_id, "annotations": target, "caption": caption}
            img, target = self.prepare(img, target)
            target["dataset_name"] = dataset_name
            for extra_key in ["sentence_id", "original_img_id", "original_id", "task_id"]:
                if extra_key in coco_img:
                    target[extra_key] = coco_img[extra_key] # box xyxy -> cxcywh
            # FIXME: handle "valid", since some box may be removed due to random crop
            target["valid"] = torch.tensor([1]) if len(target["area"]) != 0 else torch.tensor([0])

            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

            ####
            imgs = [img for i in range(frame_num)] # [PIL]
            masks = [target["masks"] for i in range(frame_num)] # [tensor [1 H W]]

            resized_images, resized_masks = [], []
            ####

            # if self.image_set == "train":
            final_scales1 = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
            scales2 = [400, 500, 600]
            final_scales2 = [296, 328, 360, 392, 416, 448, 480, 512]

            # reshape image and padding
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            transform3_1 = transforms.ColorJitter(brightness=(0.5, 1.5))
            transform3_2 = transforms.ColorJitter(contrast=(0.5, 1.5))
            transform3_3 = transforms.ColorJitter(saturation=(0.5, 1.5))
            transform3_4 = transforms.ColorJitter(hue=(-0.1, 0.1))
            transform4 = transforms.RandomHorizontalFlip(p=1)

            ####
            for img, mask in zip(imgs, masks):
                resized_image = transform(img)

                resized_images.append(resized_image)
                resized_masks.append(mask)
            ####

            random_h = random.choice(final_scales1)
            random_w = random.choice(final_scales1)
            random_h_1 = random.choice(scales2)
            random_w_1 = random.choice(scales2)
            random_h_2 = random.choice(final_scales2)
            random_w_2 = random.choice(final_scales2)
            transform5 = transforms.Resize((random_h, random_w), interpolation=Image.NEAREST)
            transform5_2 = transforms.Resize((random_h_1, random_w_1), interpolation=Image.NEAREST)
            transform5_3 = transforms.Resize((random_h_2, random_w_2), interpolation=Image.NEAREST)

            min_scale = 384
            max_scale = 600
            # random_h_3 = random.randint(min_scale, min(random_h_1, max_scale))
            # random_w_3 = random.randint(min_scale, min(random_w_1, max_scale))

            # 颜色增强
            ####
            if random.random() > 0.3:
                for index, img in enumerate(resized_images):
                    # 随机调整亮度
                    resized_images[index] = transform3_1(img)
            if random.random() > 0.3:
                for index, img in enumerate(resized_images):
                    # 随机调整对比度
                    resized_images[index] = transform3_2(img)
            if random.random() > 0.3:
                for index, img in enumerate(resized_images):
                    # 随机调整饱和度
                    resized_images[index] = transform3_3(img)
            if random.random() > 0.3:
                for index, img in enumerate(resized_images):
                    # 随机调整色调
                    resized_images[index] = transform3_4(img)
            if random.random() > 0.3:
                for index, img in enumerate(resized_images):
                    # 随机调整对比度
                    resized_images[index] = transform3_2(img)

            ## visualization
            # plt.imshow(resized_images[0].permute(1, 2, 0))
            # plt.imshow(resized_masks[2].permute(1, 2, 0), cmap="jet", alpha=0.5)

            ####
            # 随机翻转
            resized_caption = caption
            # if random.random() > 0.5:
            if False:
                for index, img in enumerate(resized_images):
                    resized_images[index] = transform4(resized_images[index])
                    resized_masks[index] = transform4(resized_masks[index])
                resized_caption = caption.replace('left', '@').replace('right', 'left').replace('@', 'right')
            # 随机大小变化
            if random.random() > 0.5:
                for index, img in enumerate(resized_images):
                    resized_images[index] = transform5(resized_images[index])
                    resized_masks[index] = transform5(resized_masks[index])
            else:
                for index, img in enumerate(resized_images):
                    resized_images[index] = transform5_2(resized_images[index])
                    resized_masks[index] = transform5_2(resized_masks[index])
                    resized_images[index], resized_masks[index] = random_crop(resized_images[index], resized_masks[index], size=(random.randint(min_scale, min(resized_images[index].shape[1], max_scale)), random.randint(min_scale, min(resized_images[index].shape[2], max_scale))))
                    resized_images[index] = transform5_3(resized_images[index])
                    resized_masks[index] = transform5_3(resized_masks[index])

            for index, img in enumerate(resized_images):
                resized_images[index] = resized_images[index].permute(1, 2, 0)
                resized_masks[index] = resized_masks[index].squeeze(0)

            #####
            resized_bbox = []
            valid = []
            # get mask correspondent bbox
            for index, mask in enumerate(resized_masks):
                if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask.numpy().astype(np.float32))
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    resized_bbox.append(box)
                    valid.append(1)
                else:
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    resized_bbox.append(box)
                    valid.append(0)
            # target['boxes'] = resized_bbox
            # target['valid'] = torch.tensor(valid)

            # show bbox
            import matplotlib.patches as patches
            # rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            # fig, ax = plt.subplots(1)
            # ax.imshow(img)
            # ax.add_patch(rect)

            target = {
                # 'labels': labels,                      # [,]
                'boxes': resized_bbox,                   # [4], xyxy
                'masks': resized_masks,                  # [H, W]
                'caption': resized_caption,              # str
                'valid': torch.tensor(valid),            # [tensor]
            }

            # FIXME: handle "valid", since some box may be removed due to random crop
            if torch.any(target['valid'] == 1):  # at leatst exist instance
                instance_check = True
            else:
                idx = random.randint(0, self.__len__() - 1)

        return resized_images, target
    
    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2



def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        # 将多边形分割转换为COCO格式的RLE编码
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        caption = target["caption"] if "caption" in target else None

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2] # xminyminwh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        # keep the valid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if caption is not None:
            target["caption"] = caption
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["valid"] = torch.tensor([1])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target


def build_refexp(dataset_file, image_set, args):
    root = Path("../coco")
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    dataset = dataset_file
    PATHS = {
        "train": (root / "train2014", root / dataset / f"{mode}_{dataset}_train.json"),
        "val": (root / "train2014", root / dataset / f"{mode}_{dataset}_val.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = ModulatedDetection(
        img_folder,
        ann_file,
        image_set=image_set,
        return_masks=True,
        args=args,
    )
    return dataset


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


# build evaluator list for dataset_val
def build_evaluator_list(base_ds, dataset_name):
    """Helper function to build the list of evaluators for a given dataset"""
    evaluator_list = []
    iou_types = ["bbox"]
    iou_types.append("segm")

    evaluator_list.append(CocoEvaluator(base_ds, tuple(iou_types), useCats=False))
    # TODO: currently ont support RefExpEvaluator (memory error)
    return evaluator_list

# 定义随机裁剪函数
def random_crop(image, mask, size):
    assert image.shape[1:] == mask.shape[1:], "图像和掩膜形状不匹配"

    _, h, w = image.shape
    th, tw = size
    if w == tw and h == th:
        return image, mask

    x = torch.randint(0, w - tw + 1, size=(1,))
    y = torch.randint(0, h - th + 1, size=(1,))
    ## 增加裁减区域大小
    # extra_w = int(0.15 * w)
    # extra_h = int(0.15 * h)
    # x = torch.randint(0, w - tw + 1 + extra_w, size=(1,))
    # y = torch.randint(0, h - th + 1 + extra_h, size=(1,))

    cropped_image = image[:, y:y + th, x:x + tw]
    cropped_mask = mask[:, y:y + th, x:x + tw]

    return cropped_image, cropped_mask