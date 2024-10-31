import numpy as np
import torch

import torch.nn as nn
from torch.nn import functional as F
import os
from tqdm import tqdm
import argparse
import warnings
from per_segment_anything import sam_model_registry, SamPredictor
from PIL import Image
from transformers import T5EncoderModel, T5TokenizerFast
import json
import logging
from torch.utils.data import Dataset, DataLoader
from datasets.categories import ytvos_category_dict as category_dict
import torchvision.transforms as transforms
import util.misc as utils
import datasets.samplers as samplers
import torch.distributed as dist
from einops import rearrange

import opts
import refersam
import loss

import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
warnings.filterwarnings('ignore')


class Ref_ytvos(Dataset):

    def __init__(self, img_folder, ann_file, args=[]):
        self.img_folder = img_folder
        self.ann_file = ann_file

        # create video meta data
        self.prepare_metas()
        self.args = args

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))
        print('\n')

    def prepare_metas(self):
        # read object information
        with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']

        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len, 1):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['obj_id'] = int(exp_dict['obj_id'])
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    # get object category
                    obj_id = exp_dict['obj_id']
                    meta['category'] = vid_meta['objects'][obj_id]['category']
                    # get split expression
                    self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        instance_check = False
        # set default img size

        while not instance_check:
            meta = self.metas[idx]  # dict

            video, exp, obj_id, category, frames, frame_id = \
                        meta['video'], meta['exp'], meta['obj_id'], meta['category'], meta['frames'], meta['frame_id']

            # clean up the caption
            exp = " ".join(exp.lower().split())
            category_id = category_dict[category]
            vid_len = len(frames)

            # use memory frame
            num_frames = self.args.frame_num
            # random sparse sample
            sample_indx = [frame_id]
            if self.args.frame_num != 1:
                # local sample
                sample_id_before = random.randint(1, 3)
                sample_id_after = random.randint(1, 3)
                local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                # local_indx is image in one video's index
                sample_indx.extend(local_indx)

                # global sampling
                if num_frames > 3:
                    all_inds = list(range(vid_len))
                    global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                    global_n = num_frames - len(sample_indx)
                    if len(global_inds) > global_n:
                        select_id = random.sample(range(len(global_inds)), global_n)
                        for s_id in select_id:
                            sample_indx.append(global_inds[s_id])
                    elif vid_len >=global_n:  # sample long range global frames
                        select_id = random.sample(range(vid_len), global_n)
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
                    else:
                        select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))
                        for s_id in select_id:
                            sample_indx.append(all_inds[s_id])
            sample_indx.sort()

            # read frames and masks
            imgs, labels, boxes, masks, valid = [], [], [], [], []
            for j in range(self.args.frame_num):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                # img = cv2.imread(img_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # [w h c]
                mask = Image.open(mask_path).convert('P')

                # create the target
                label = torch.tensor(category_id)
                mask = np.array(mask)
                mask = (mask==obj_id).astype(np.float32) # 0,1 binary

                # reshape image and padding
                img = Image.open(img_path)

                # append
                imgs.append(img)
                labels.append(label)
                masks.append(mask)

            # image to numpy # [w h c]
            # transform
            target = {
                'labels': labels,                        # [,]
                # 'boxes': box,                          # [4], xyxy
                'masks': masks,                  # [H, W]
                'caption': exp,
                # 'valid': torch.tensor(valid),
            }

            resized_images = []
            resized_masks = []

            final_scales = [288, 320, 352, 392, 416, 448, 480, 512]
            final_scales_2 = [400, 500, 600]

            # reshape image and padding
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            transform3_1 = transforms.ColorJitter(brightness=(0.5, 1.5))
            transform3_2 = transforms.ColorJitter(contrast=(0.5, 1.5))
            transform3_3 = transforms.ColorJitter(saturation=(0.5, 1.5))
            transform3_4 = transforms.ColorJitter(hue=(-0.1, 0.1))
            transform4 = transforms.RandomHorizontalFlip(p=1)

            for img,mask in zip(imgs, masks):
                resized_image = transform(img)  # [3 640 640]
                resized_mask = transform(mask)
                resized_images.append(resized_image)
                resized_masks.append(resized_mask)

            random_h = random.choice(final_scales)
            random_w = random.choice(final_scales)
            random_h_2 = random.choice(final_scales_2)
            random_w_2 = random.choice(final_scales_2)
            transform5 = transforms.Resize((random_h, random_w), interpolation=Image.NEAREST)
            transform5_2 = transforms.Resize((random_h_2, random_w_2), interpolation=Image.NEAREST)

            min_scale = 384
            max_scale = 600
            random_h_3 = random.randint(min_scale, min(random_h_2, max_scale))
            random_w_3 = random.randint(min_scale, min(random_w_2, max_scale))

            # 颜色增强
            if random.random() > 0.3:
                for index,img in enumerate(resized_images):
                    # 随机调整亮度
                    resized_images[index] = transform3_1(img)
            if random.random() > 0.3:
                for index,img in enumerate(resized_images):
                    # 随机调整对比度
                    resized_images[index] = transform3_2(img)
            if random.random() > 0.3:
                for index,img in enumerate(resized_images):
                    # 随机调整饱和度
                    resized_images[index] = transform3_3(img)
            if random.random() > 0.3:
                for index,img in enumerate(resized_images):
                    # 随机调整色调
                    resized_images[index] = transform3_4(img)
            if random.random() > 0.3:
                for index,img in enumerate(resized_images):
                    # 随机调整对比度
                    resized_images[index] = transform3_2(img)

            # 随机翻转
            if random.random() > 0.5:
                for index, img in enumerate(resized_images):
                    resized_images[index] = transform4(resized_images[index])
                    resized_masks[index] = transform4(resized_masks[index])
                target['caption'] = exp.replace('left', '@').replace('right', 'left').replace('@', 'right')

            # 随机大小变化
            if random.random() > 0.5:
                for index, img in enumerate(resized_images):
                    resized_images[index] = transform5(resized_images[index])
                    resized_masks[index] = transform5(resized_masks[index])
            else:
                for index, img in enumerate(resized_images):
                    resized_images[index] = transform5_2(resized_images[index])
                    resized_masks[index] = transform5_2(resized_masks[index])
                    resized_images[index], resized_masks[index] = random_crop(resized_images[index], resized_masks[index], size=(random_h_3, random_w_3))
                    resized_images[index] = transform5(resized_images[index])
                    resized_masks[index] = transform5(resized_masks[index])

            for index, img in enumerate(resized_images):
                resized_images[index] = resized_images[index].permute(1, 2, 0)
                resized_masks[index] = resized_masks[index].squeeze(0)
            target['masks'] = resized_masks

            resized_bbox = []
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
            target['boxes'] = resized_bbox
            target['valid'] = torch.tensor(valid)

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


# 定义随机裁剪函数
def random_crop(image, mask, size):
    assert image.shape[1:] == mask.shape[1:], "图像和掩膜形状不匹配"

    _, h, w = image.shape
    th, tw = size
    if w == tw and h == th:
        return image, mask

    x = torch.randint(0, w - tw + 1, size=(1,))
    y = torch.randint(0, h - th + 1, size=(1,))

    cropped_image = image[:, y:y + th, x:x + tw]
    cropped_mask = mask[:, y:y + th, x:x + tw]

    return cropped_image, cropped_mask
