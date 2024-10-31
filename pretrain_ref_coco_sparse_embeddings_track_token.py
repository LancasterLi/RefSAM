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
import matplotlib.pyplot as plot

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

            ## use one frame generate multi frames
            num_frames = self.args.num_frames

            if self.image_set == "train":
                scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
                scales2 = [400, 500, 600]
                final_scales = [296, 328, 360, 392, 416, 448, 480, 512]

                # reshape image and padding
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
                # transform3 = transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5),
                #                                     hue=(-0.1, 0.1))
                transform3_1 = transforms.ColorJitter(brightness=(0.5, 1.5))
                transform3_2 = transforms.ColorJitter(contrast=(0.5, 1.5))
                transform3_3 = transforms.ColorJitter(saturation=(0.5, 1.5))
                transform3_4 = transforms.ColorJitter(hue=(-0.1, 0.1))
                transform4 = transforms.RandomHorizontalFlip(p=1)

                resized_images = transform(img) # [3 640 640] tensor
                resized_masks = target['masks']

                random_h = random.choice(scales)
                random_w = random.choice(scales)
                random_h_1 = random.choice(scales2)
                random_w_1 = random.choice(scales2)
                random_h_2 = random.choice(final_scales)
                random_w_2 = random.choice(final_scales)
                # transform5 = transforms.RandomResizedCrop((random_h, random_w), scale=(0.5, 1.5))
                transform5 = transforms.Resize((random_h, random_w), interpolation=Image.NEAREST)
                transform5_2 = transforms.Resize((random_h_1, random_w_1), interpolation=Image.NEAREST)
                transform5_3 = transforms.Resize((random_h_2, random_w_2), interpolation=Image.NEAREST)

                # transform6 = transforms.RandomResizedCrop(size=(384, 600), scale=(0.8, 1.0), ratio=(0.75, 1.333))
                min_scale = 384
                max_scale = 600
                random_h_3 = random.randint(min_scale, min(random_h_1, max_scale))
                random_w_3 = random.randint(min_scale, min(random_w_1, max_scale))
                transform6 = transforms.RandomCrop(size=(random_h_3, random_w_3))

                ## use one frame generate multi frames
                all_resized_images = [] # [N C H W]
                all_resized_masks = [] # [N 1 H W]
                for i in range(num_frames):
                    all_resized_images.append(resized_images)
                    all_resized_masks.append(resized_masks)
                valid = target['valid'].repeat(num_frames)
                target['valid'] = valid
                imgs, masks = [], []

                # 颜色增强
                if random.random() > 0.3:
                    for index, img in enumerate(all_resized_images):
                        # 随机调整亮度
                        all_resized_images[index] = transform3_1(img)
                if random.random() > 0.3:
                    for index, img in enumerate(all_resized_images):
                        # 随机调整对比度
                        all_resized_images[index] = transform3_2(img)
                if random.random() > 0.3:
                    for index, img in enumerate(all_resized_images):
                        # 随机调整饱和度
                        all_resized_images[index] = transform3_3(img)
                if random.random() > 0.3:
                    for index, img in enumerate(all_resized_images):
                        # 随机调整色调
                        all_resized_images[index] = transform3_4(img)
                if random.random() > 0.3:
                    for index, img in enumerate(all_resized_images):
                        # 随机调整对比度
                        all_resized_images[index] = transform3_2(img)

                # 随机翻转
                if random.random() > 0.5:
                    for index, img in enumerate(all_resized_images):
                        all_resized_images[index] = transform4(all_resized_images[index])
                        all_resized_masks[index] = transform4(all_resized_masks[index])
                    target['caption'] = caption.replace('left', '@').replace('right', 'left').replace('@', 'right')
                # 随机大小变化
                if random.random() > 0.5:
                    for index, img in enumerate(all_resized_images):
                        all_resized_images[index] = transform5(all_resized_images[index])
                        all_resized_masks[index] = transform5(all_resized_masks[index])
                else:
                    for index, img in enumerate(all_resized_images):
                        all_resized_images[index] = transform5_2(all_resized_images[index])
                        all_resized_masks[index] = transform5_2(all_resized_masks[index])
                        # all_resized_images[index] = transform6(all_resized_images[index])
                        # all_resized_masks[index] = transform6(all_resized_masks[index])
                        all_resized_images[index], all_resized_masks[index] = random_crop(all_resized_images[index], all_resized_masks[index], size=(random_h_3, random_w_3))
                        all_resized_images[index] = transform5_3(all_resized_images[index])
                        all_resized_masks[index] = transform5_3(all_resized_masks[index])

                # t5 = transforms.Resize((max_height, max_width))
                # resized_images = t5(resized_images)
                # resized_masks = t5(resized_masks)

                for index, img in enumerate(all_resized_images):
                    all_resized_images[index] = all_resized_images[index].permute(1, 2, 0)
                    all_resized_masks[index] = all_resized_masks[index].squeeze(0)
                target['masks'] = all_resized_masks
                resized_images = all_resized_images
            else:
                # reshape image and padding
                final_scales = [296, 328, 360, 392, 416, 448, 480, 512]
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
                random_h = random.choice(final_scales)
                random_w = random.choice(final_scales)
                transform5 = transforms.Resize((random_h, random_w), interpolation=Image.NEAREST)

                resized_images = transform(img)
                resized_masks = target['masks']
                # 随机大小变化
                if random.random() > 0.5:
                    resized_images = transform5(resized_images)
                    resized_masks = transform5(resized_masks)

                resized_images = resized_images.permute(1, 2, 0) # [640 640 3]
                resized_masks = resized_masks.squeeze(0) # [640 640]

                target['masks'] = resized_masks

        return resized_images, target


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
    root = Path("coco")
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


def main():
    # init opts
    args = opts.get_arguments()
    print("Args:", args)

    # create output path
    output_path = './outputs/' + args.outdir
    if not os.path.exists('outputs-before/'):
        os.mkdir('outputs-before/')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # init logger
    if args.train_decoder:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=os.path.join(output_path, str(args.lr_decoder) + 'decoder_log.txt'),
            filemode='a',
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=os.path.join(output_path, 'log.txt'),
            filemode='a',
        )

    logger = logging.getLogger()
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    logger.info('arguments:')
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    # init distribute
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # load model
    if args.distributed and dist.get_rank() == 0:
        logger.info("======> load model")
    text_model_name = args.text_encoder
    model = refersam.Model(args, text_model_name, logger).to('cuda')
    # model = nn.DataParallel(model)
    model.to(device)
    # logger.info(f'model structure: {model}')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True) # for use memory
        model_without_ddp = model.module

    # set param groups
    param_groups = []
    add_to_resizer_param_groups = []
    add_to_mask_decoder_param_groups = []
    add_to_dense_conv_param_groups = []
    add_to_lora_param_groups = []
    add_to_memory_mask_conv_groups = []

    add_to_track_query_attn = []
    for name, param in model_without_ddp.named_parameters():
        if "resizer" in name:
            param.requires_grad = True
            add_to_resizer_param_groups.append(param)
        elif "dense" in name or "sparse_fc" in name: # for dense_conv and sparse_fc
            param.requires_grad = True
            add_to_dense_conv_param_groups.append(param)
        elif args.train_decoder and 'sam.mask_decoder' in name:
            param.requires_grad = True
            add_to_mask_decoder_param_groups.append(param)
        elif args.train_image_encoder_lora and ('w_a' in name or 'w_b' in name or 'w_a_qkv' in name or 'w_b_qkv' in name):
            param.requires_grad = True
            add_to_lora_param_groups.append(param)        
        elif 'memory_key' in name or 'memory_value' in name:
            param.requires_grad = True
            add_to_memory_mask_conv_groups.append(param)
        elif 'temporal_aggregation_network' in name:
            param.requires_grad = True
            add_to_track_query_attn.append(param)
        else:
            param.requires_grad = False
    param_groups.append(
        {
            "params": add_to_resizer_param_groups, "lr": args.lr
        }
    )
    if len(add_to_memory_mask_conv_groups) != 0:
        param_groups.append(
            {
                'params': add_to_memory_mask_conv_groups, 'lr': args.lr_memory
            }
        )
    if args.train_decoder:
        param_groups.append(
            {
                'params': add_to_mask_decoder_param_groups, 'lr': args.lr_decoder
            }
        )
    if args.spatial_dynamic_fusion:
        param_groups.append(
            {
                'params': add_to_dense_conv_param_groups, 'lr': args.lr_dense_conv
            }
        )
    if args.train_image_encoder_lora:
        param_groups.append(
            {
                'params': add_to_lora_param_groups, 'lr': args.lr_image_encoder_lora
            }
        )
    if args.track_query_attn:
        param_groups.append(
            {
                'params': add_to_track_query_attn, 'lr': args.lr_track_query_attn
            }
        )

    # check trained parameters
    logger.info("trained parameters:\n")
    for name, param in model_without_ddp.named_parameters():
        if param.requires_grad:
            logger.info(name)

    if args.distributed and dist.get_rank() == 0:
        n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        logger.info('number of params:', n_parameters)

        logger.info("======> train config")
    # load train dataset
    data_file = "all"
    dataset_names = ["refcoco", "refcoco+", "refcocog"]
    train_dataset = torch.utils.data.ConcatDataset(
        [build_refexp(name, image_set="train", args=args) for name in dataset_names]
    )

    if args.distributed and dist.get_rank() == 0:
        logger.info("\nTrain dataset sample number: ", len(train_dataset))
        logger.info("\n")

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(train_dataset)
        else:
            sampler_train = samplers.DistributedSampler(train_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset) #02
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn_track_memory, num_workers=args.num_workers)

    # load valid dataset
    Val_all = namedtuple(typename="val_data", field_names=["dataset_name", "dataloader", "base_ds", "evaluator_list"])
    val_tuples = []

    for name in dataset_names:
        val_dataset = build_refexp(name, image_set="val", args=args)
        sampler_val = (
            samplers.DistributedSampler(val_dataset, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(val_dataset)
        )
        data_loader_val = DataLoader(
            val_dataset,
            args.batch_size,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers
        )
        base_ds = get_coco_api_from_dataset(val_dataset)
        val_tuples.append(Val_all(dataset_name=name, dataloader=data_loader_val, base_ds=base_ds, evaluator_list=None))

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch)
    if args.distributed and dist.get_rank() == 0:
        logger.info("======> staring training")
    for train_idx in tqdm(range(args.train_epoch)):
        if args.distributed:
            sampler_train.set_epoch(train_idx)

        # model.module.resizer.train()
        model.train()
        one_epoch_loss = 0
        sampler_num = 0

        for idx, (img, mask, caption, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            # target['img'] = img

            img = img.to(device)
            mask = mask.to(device)

            img = rearrange(img, "b n h w c -> n b h w c")
            mask = rearrange(mask, "b n h w -> n b h w")
            # rearrange(origin_key, "b c h w -> (b h w) c")

            ## each perform one time frame, update track query
            track_tokens_list = [] # for first frame
            for i in range(img.shape[1]):
                track_tokens_list.append(None)
            use_track_tokens = True
            # use_track_tokens = False

            one_iter_sampler = 0
            one_loss = 0
            for one_time_img, one_time_mask in zip(img, mask):
                # for no token
                for i in range(img.shape[1]):
                    track_tokens_list.append(None)

                train_mask_list = one_time_mask

                high_res_masks_list, track_tokens_list = model(one_time_img, caption, target, track_tokens_list, use_track_tokens) ## None is track tokens

                if isinstance(high_res_masks_list, list):
                    high_res_masks_list = torch.cat(high_res_masks_list, dim=0)

                for high_res_masks, train_mask in zip(high_res_masks_list, train_mask_list):
                    high_res_masks = high_res_masks.flatten(1)

                    gt_mask = train_mask
                    if isinstance(gt_mask, torch.Tensor):
                        gt_mask = gt_mask.cpu().float().unsqueeze(0).flatten(1).cuda()
                    else:
                        gt_mask = torch.Tensor(gt_mask.cpu()).float().unsqueeze(0).flatten(1).cuda()

                    dice_loss = loss.calculate_dice_loss(high_res_masks, gt_mask)  # 用于图像分割中的目标边界分割 < 0.1较好
                    focal_loss = loss.calculate_sigmoid_focal_loss(high_res_masks, gt_mask)  # 用于像素级别分类
                    # point_loss = loss_fn(after_text_sentence_feature, target_feat)
                    one_loss += dice_loss + focal_loss
                    one_iter_sampler += 1

            one_loss /= one_iter_sampler
            sampler_num += 1
            one_epoch_loss += one_loss.item()
            one_loss.backward()
            optimizer.step()

            if (args.distributed and dist.get_rank() == 0) or not args.distributed:
                if idx % 10 == 0:
                    logger.info("index:{}, total_sum:{}, loss:{}, lr:{}".format(idx, len(train_dataloader), one_loss, args.lr))

        # scheduler.step()

        one_epoch_loss /= sampler_num
        # logger.info("loss_sum:{}".format(loss))
        if (args.distributed and dist.get_rank() == 0) or not args.distributed:
            if train_idx % args.log_epoch == 0:
                logger.info('Train Epoch: {:} / {:}'.format(train_idx, args.train_epoch))
                # current_lr = scheduler.get_last_lr()[0]
                # logger.info('LR: {:.6f}, Loss: {:.4f}'.format(current_lr, one_epoch_loss))
                logger.info('LR: {:.6f}, Loss: {:.4f}'.format(args.lr, one_epoch_loss))

        if (args.distributed and dist.get_rank() == 0) or not args.distributed:
            # save last model
            torch.save(model_without_ddp.resizer.state_dict(),
                       os.path.join(output_path, 'resizer_' + text_model_name + '_last.pth'))
            if args.train_decoder:
                torch.save(model_without_ddp.sam.mask_decoder.state_dict(),
                           os.path.join(output_path, 'mask_decoder_' + text_model_name + '_last.pth'))
            if args.spatial_dynamic_fusion:
                torch.save(model_without_ddp.dense_conv.state_dict(),
                           os.path.join(output_path, 'dense_conv_' + text_model_name + '_last.pth'))
            if args.sparse_attention:
                torch.save(model_without_ddp.sparse_fc.state_dict(),
                           os.path.join(output_path, 'sparse_fc' + text_model_name + '_last.pth'))
            if args.train_image_encoder_lora:
                torch.save(model_without_ddp.sam.image_encoder.blocks.state_dict(),
                           os.path.join(output_path, 'image_encoder_blocks_' + text_model_name + '_last.pth'))
            if args.track_query_attn:
                torch.save(model_without_ddp.temporal_aggregation_network.state_dict(),
                           os.path.join(output_path, 'track_net_' + text_model_name + '_last.pth'))
            if args.word_memory:
                torch.save(model_without_ddp.memory_key.state_dict(),
                           os.path.join(output_path, 'memory_key_' + text_model_name + '_last.pth'))
                torch.save(model_without_ddp.memory_value.state_dict(),
                           os.path.join(output_path, 'memory_value_' + text_model_name + '_last.pth'))
 
            # each N epoch save one model
            if train_idx % 1 == 0:
                torch.save(model_without_ddp.resizer.state_dict(), os.path.join(output_path, 'resizer_' + text_model_name + "_" + str(train_idx) + '.pth'))
                if args.train_decoder:
                    torch.save(model_without_ddp.sam.mask_decoder.state_dict(), os.path.join(output_path, 'mask_decoder_' + text_model_name + "_" + str(train_idx) + '.pth'))
                if args.spatial_dynamic_fusion:
                    torch.save(model_without_ddp.dense_conv.state_dict(), os.path.join(output_path, 'dense_conv_' + text_model_name + "_" + str(train_idx) + '.pth'))
                if args.sparse_attention:
                    torch.save(model_without_ddp.sparse_fc.state_dict(), os.path.join(output_path, 'sparse_fc' + text_model_name + "_" + str(train_idx) + '.pth'))
                if args.train_image_encoder_lora:
                    torch.save(model_without_ddp.sam.image_encoder.blocks.state_dict(),
                               os.path.join(output_path, 'image_encoder_blocks_' + text_model_name + "_" + str(train_idx) + '.pth'))
                if args.track_query_attn:
                    torch.save(model_without_ddp.temporal_aggregation_network.state_dict(),
                               os.path.join(output_path, 'track_net_' + text_model_name + "_" + str(train_idx) + '.pth'))
                if args.word_memory:
                    torch.save(model_without_ddp.memory_key.state_dict(),
                            os.path.join(output_path, 'memory_key_' + text_model_name + "_" + str(train_idx) + '.pth'))
                    torch.save(model_without_ddp.memory_value.state_dict(),
                            os.path.join(output_path, 'memory_value_' + text_model_name + "_" + str(train_idx) + '.pth'))


        # each epoch eval
        test_stats = {}
        for i, item in enumerate(val_tuples):
            evaluator_list = build_evaluator_list(item.base_ds, item.dataset_name)
            item = item._replace(evaluator_list=evaluator_list)
            logger.info(f"\n Evaluating {item.dataset_name}")

            model.eval()
            one_epoch_loss = 0
            sampler_num = 0

            for idx, (img, mask, caption, target) in enumerate(item.dataloader):
                ## each perform one time frame, update track query
                track_tokens_list = []  # for first frame
                for i in range(img.shape[1]):
                    track_tokens_list.append(None)
                use_track_tokens = True
                # use_track_tokens = False

                img = img.to(device)
                mask = mask.to(device)
                train_mask_list = mask

                with torch.no_grad():
                    # high_res_masks_list = model(target)
                    high_res_masks_list, _ = model(img, caption, target, track_tokens_list, use_track_tokens)

                one_iter_sampler = 0
                one_loss = 0
                if isinstance(high_res_masks_list, list):
                    high_res_masks_list = torch.cat(high_res_masks_list, dim=0)

                for high_res_masks, train_mask in zip(high_res_masks_list, train_mask_list):
                    high_res_masks = high_res_masks.flatten(1)

                    gt_mask = train_mask
                    if isinstance(gt_mask, torch.Tensor):
                        gt_mask = gt_mask.cpu().float().unsqueeze(0).flatten(1).cuda()
                    else:
                        gt_mask = torch.Tensor(gt_mask.cpu()).float().unsqueeze(0).flatten(1).cuda()

                    dice_loss = loss.calculate_dice_loss(high_res_masks, gt_mask)  
                    focal_loss = loss.calculate_sigmoid_focal_loss(high_res_masks, gt_mask)  
                    one_loss += dice_loss + focal_loss
                    one_iter_sampler += 1

                one_loss /= one_iter_sampler
                sampler_num += 1
                one_epoch_loss += one_loss.item()                

                if idx % 100 == 0:
                    logger.info("eval index:{}, total_sum:{}, loss:{}".format(idx, len(item.dataloader), one_loss))

            one_epoch_loss /= sampler_num

            if train_idx % args.log_epoch == 0:
                logger.info('Train Epoch: {:} / {:}'.format(train_idx, args.train_epoch))
                logger.info('eval dataset {} Loss: {:.4f}'.format(item.dataset_name, one_epoch_loss))

    if not args.distributed or dist.get_rank() == 0:
        torch.save(model_without_ddp.resizer.state_dict(),
                   os.path.join(output_path, 'resizer_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
        if args.train_decoder:
            torch.save(model_without_ddp.sam.mask_decoder.state_dict(),
                       os.path.join(output_path, 'mask_decoder_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
        if args.spatial_dynamic_fusion:
            torch.save(model_without_ddp.dense_conv.state_dict(),
                       os.path.join(output_path, "dense_conv_" + text_model_name + "_" + str(args.train_epoch) + '.pth'))
        if args.sparse_attention:
            torch.save(model_without_ddp.sparse_fc.state_dict(), 
                       os.path.join(output_path, 'sparse_fc' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
        if args.train_image_encoder_lora:
            torch.save(model_without_ddp.sam.image_encoder.blocks.state_dict(),
                       os.path.join(output_path, 'image_encoder_blocks_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
        if args.word_memory:
            torch.save(model_without_ddp.memory_key.state_dict(),
                    os.path.join(output_path, 'memory_key_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
            torch.save(model_without_ddp.memory_value.state_dict(),
                    os.path.join(output_path, 'memory_value_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
        if args.track_query_attn:
            torch.save(model_without_ddp.temporal_aggregation_network.state_dict(),
                       os.path.join(output_path, 'track_net_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
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

    cropped_image = image[:, y:y + th, x:x + tw]
    cropped_mask = mask[:, y:y + th, x:x + tw]

    return cropped_image, cropped_mask


if __name__ == "__main__":
    main()
