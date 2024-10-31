import numpy as np
import torch

import torch.nn as nn
from torch.nn import functional as F
import os
from tqdm import tqdm
import argparse
import warnings
from PIL import Image
import json
import logging
from torch.utils.data import Dataset, DataLoader
from datasets.categories import ytvos_category_dict as category_dict
import torchvision.transforms as transforms
import util.misc as utils
import datasets.samplers as samplers
import torch.distributed as dist
from einops import rearrange
import matplotlib.pyplot as plt

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
            num_frames = self.args.num_frames
            # random sparse sample
            sample_indx = [frame_id]
            if self.args.num_frames != 1:
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
            for j in range(self.args.num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                mask = Image.open(mask_path).convert('P')

                # create the target
                label = torch.tensor(category_id)
                mask = np.array(mask)
                mask = (mask==obj_id).astype(np.float32) # 0,1 binary
                if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    # box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else: # some frame didn't contain the instance
                    # box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                    valid.append(0)

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
                'valid': torch.tensor(valid),
            }

            ######################################### 数据增强
            resized_images = []
            resized_masks = []

            # final_scales = [296, 328, 360, 392, 416, 448, 480, 512]
            final_scales = [288, 320, 352, 392, 416, 448, 480, 512]
            final_scales_2 = [400, 500, 600]

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

            for img,mask in zip(imgs, masks):
                resized_image = transform(img)  # [3 640 640]
                resized_mask = transform(mask)
                resized_images.append(resized_image)
                resized_masks.append(resized_mask)

            random_h = random.choice(final_scales)
            random_w = random.choice(final_scales)
            random_h_2 = random.choice(final_scales_2)
            random_w_2 = random.choice(final_scales_2)
            # transform5 = transforms.RandomResizedCrop((random_h, random_w), scale=(0.5, 1.5))
            transform5 = transforms.Resize((random_h, random_w), interpolation=Image.NEAREST)
            transform5_2 = transforms.Resize((random_h_2, random_w_2), interpolation=Image.NEAREST)

            # transform6 = transforms.RandomResizedCrop(size=(384, 600), scale=(0.8, 1.0), ratio=(0.75, 1.333))
            min_scale = 384
            max_scale = 600
            random_h_3 = random.randint(min_scale, min(random_h_2, max_scale))
            random_w_3 = random.randint(min_scale, min(random_w_2, max_scale))
            transform6 = transforms.RandomCrop(size=(random_h_3, random_w_3))

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
                    # all_resized_images[index] = transform6(all_resized_images[index])
                    # all_resized_masks[index] = transform6(all_resized_masks[index])
                    resized_images[index], resized_masks[index] = random_crop(resized_images[index], resized_masks[index], size=(random_h_3, random_w_3))
                    resized_images[index] = transform5(resized_images[index])
                    resized_masks[index] = transform5(resized_masks[index])

            # t5 = transforms.Resize((max_height, max_width))
            # resized_images = t5(resized_images)
            # resized_masks = t5(resized_masks)
            for index, img in enumerate(resized_images):
                resized_images[index] = resized_images[index].permute(1, 2, 0)
                resized_masks[index] = resized_masks[index].squeeze(0)
            target['masks'] = resized_masks
            ######################################### 数据增强

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


def main():
    # init opts
    args = opts.get_arguments()
    args.data = './ref-youtube-vos/train'
    # args.outdir = './ref_ytvos_sparse_dense_embeddings'
    print("Args:", args)

    # create output path
    output_path = './outputs/' + args.outdir
    if not os.path.exists('outputs-before/'):
        os.mkdir('outputs-before/')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # set log txt
    # init logger
    if args.train_decoder:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=os.path.join(output_path, str(args.lr_decoder) + '_decoder_log.txt'),
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

    # load pretrain model
    if args.pretrain:
        if args.proj_mlp:
            model.resizer.load_state_dict(torch.load(args.pretrain_mlp))
        if args.dense_embeddings:
            model.dense_conv.load_state_dict(torch.load(args.pretrain_dense_conv))
        if args.train_decoder:
            model.sam.mask_decoder.load_state_dict(torch.load(args.pretrain_decoder))
        if args.mask_word_memory or args.mask_memory:
            model.memory_key.load_state_dict(torch.load(args.pretrain_memory_key))
            model.memory_value.load_state_dict(torch.load(args.pretrain_memory_value))
        if args.track_query_attn and args.pretrain_track_query_attn is not None:
            model.temporal_aggregation_network.load_state_dict(torch.load(args.pretrain_track_query_attn))

    current_epoch = 0
    # resume checkpoint
    if args.resume:
        if args.proj_mlp:
            model.resizer.load_state_dict(torch.load(args.resume_mlp))
        if args.dense_embeddings:
            model.dense_conv.load_state_dict(torch.load(args.resume_dense_conv))
        if args.train_decoder:
            model.sam.mask_decoder.load_state_dict(torch.load(args.resume_decoder))
        if args.train_image_encoder_lora:
            model.sam.image_encoder.blocks.load_state_dict(torch.load(args.resume_lora_blocks))
        if args.track_query_attn:
            model.temporal_aggregation_network.load_state_dict(torch.load(args.resume_track_query_attn))
        current_epoch = args.resume_epoch

    model.to(device)

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
    ##
    add_to_multi_scale_conv = []
    add_to_track_query_attn = []
    for name, param in model_without_ddp.named_parameters():
        if "resizer" in name:
            param.requires_grad = True
            add_to_resizer_param_groups.append(param)
        elif "dense" in name: # for dense_conv
            param.requires_grad = True
            add_to_dense_conv_param_groups.append(param)
        elif args.train_decoder and 'sam.mask_decoder' in name:
            param.requires_grad = True
            add_to_mask_decoder_param_groups.append(param)
        elif args.train_image_encoder_lora and ('w_a' in name or 'w_b' in name):
            param.requires_grad = True
            add_to_lora_param_groups.append(param)
        elif 'memory_key' in name or 'memory_value' in name:
            param.requires_grad = True
            add_to_memory_mask_conv_groups.append(param)      
        elif 'mask_conv' in name:
            param.requires_grad = True
            add_to_memory_mask_conv_groups.append(param)
        elif 'multi_scale_weight' in name:
            param.requires_grad = True
            add_to_multi_scale_conv.append(param)
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
    if args.multi_scale:
        param_groups.append(
            {
                'params': add_to_memory_mask_conv_groups, 'lr': args.lr_multi_scale
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
        logger.info('trainable number of params:', n_parameters)

        n_parameters2 = sum(p.numel() for p in model_without_ddp.parameters())
        logger.info('total number of params:', n_parameters2)

        n_parameters3 = sum(p.numel() for p in model_without_ddp.sam.parameters())
        logger.info('total number of sam params:', n_parameters3)

        logger.info("======> train config")
    # load dataset
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    train_dataset = Ref_ytvos(img_folder="ref-youtube-vos/train",
                              ann_file="ref-youtube-vos/meta_expressions/train/meta_expressions.json",
                              args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(train_dataset)
        else:
            sampler_train = samplers.DistributedSampler(train_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn_track_memory, num_workers=args.num_workers)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch)
    if args.distributed and dist.get_rank() == 0:
        logger.info("======> staring training")
    if args.resume is False:
        current_epoch = -1
    for train_idx in tqdm(range(current_epoch+1, args.train_epoch)):
        if args.distributed:
            sampler_train.set_epoch(train_idx)

        # model.module.resizer.train()
        model.train()
        one_epoch_loss = 0
        sampler_num = 0

        for idx, (img, mask, caption, target) in enumerate(train_dataloader):
            optimizer.zero_grad()

            img = img.to(device) # [b n h w c]
            mask = mask.to(device) # [b n h w]

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
                    gt_mask = torch.Tensor(gt_mask.cpu()).float().unsqueeze(0).flatten(1).cuda()

                    dice_loss = loss.calculate_dice_loss(high_res_masks, gt_mask)  # 用于图像分割中的目标边界分割 < 0.1较好
                    focal_loss = loss.calculate_sigmoid_focal_loss(high_res_masks, gt_mask)  # 用于像素级别分类
                    # point_loss = loss_fn(after_text_sentence_feature, target_feat)
                    one_loss += dice_loss + focal_loss
                    one_iter_sampler += 1

            ## each time update parameters
            # 后续可以测试每个iteration update parameters
            one_loss /= one_iter_sampler
            sampler_num += 1
            one_epoch_loss += one_loss.item()
            one_loss.backward()
            optimizer.step()

            if (args.distributed and dist.get_rank() == 0) or not args.distributed:
                # for save iteration checkpoint
                if args.save_iterval:
                    if idx % args.save_iterval_num == 0:
                        if (args.distributed and dist.get_rank() == 0) or not args.distributed:
                            torch.save(model_without_ddp.resizer.state_dict(),
                                    os.path.join(output_path, 'resizer_' + text_model_name + "_" + str(train_idx) + "_iteration_" + str(idx) + '.pth'))
                            if args.train_decoder:
                                torch.save(model_without_ddp.sam.mask_decoder.state_dict(),
                                        os.path.join(output_path, 'mask_decoder_' + text_model_name + "_" + str(train_idx) + "_iteration_" + str(idx) + '.pth'))
                            if args.spatial_dynamic_fusion:
                                torch.save(model_without_ddp.dense_conv.state_dict(),
                                        os.path.join(output_path, 'dense_conv_' + text_model_name + "_" + str(train_idx) + "_iteration_" + str(idx) + '.pth'))
                            if args.train_image_encoder_lora:
                                torch.save(model_without_ddp.sam.image_encoder.blocks.state_dict(),
                                        os.path.join(output_path, 'image_encoder_blocks_' + text_model_name + "_" + str(train_idx) + "_iteration_" + str(idx) + '.pth'))
                            if args.mask_word_memory:
                                torch.save(model_without_ddp.memory_key.state_dict(),
                                        os.path.join(output_path, 'memory_key_' + text_model_name + "_" + str(train_idx) + "_iteration_" + str(idx) + '.pth'))
                                torch.save(model_without_ddp.memory_value.state_dict(),
                                        os.path.join(output_path, 'memory_value_' + text_model_name + "_" + str(train_idx) + "_iteration_" + str(idx) + '.pth'))
                            if args.mask_memory:
                                torch.save(model_without_ddp.memory_key.state_dict(),
                                        os.path.join(output_path, 'memory_key_' + text_model_name + "_" + str(train_idx) + "_iteration_" + str(idx) + '.pth'))
                                torch.save(model_without_ddp.memory_value.state_dict(),
                                        os.path.join(output_path, 'memory_value_' + text_model_name + "_" + str(train_idx) + "_iteration_" + str(idx) + '.pth'))
                                torch.save(model_without_ddp.mask_conv.state_dict(),
                                        os.path.join(output_path, 'mask_conv_' + text_model_name + "_" + str(train_idx) + "_iteration_" + str(idx) + '.pth'))
                            if args.word_memory:
                                torch.save(model_without_ddp.memory_key.state_dict(),
                                        os.path.join(output_path, 'memory_key_' + text_model_name + "_" + str(train_idx) + "_iteration_" + str(idx) + '.pth'))
                                torch.save(model_without_ddp.memory_value.state_dict(),
                                        os.path.join(output_path, 'memory_value_' + text_model_name + "_" + str(train_idx) + "_iteration_" + str(idx) + '.pth'))
                            if args.track_query_attn:
                                torch.save(model_without_ddp.temporal_aggregation_network.state_dict(), os.path.join(output_path, 'track_net_' + text_model_name + "_" + str(train_idx) + "_iteration_" + str(idx) + '.pth'))

                if idx % 10 == 0:
                    logger.info("index:{}, total_sum:{}, loss:{}, lr:{}".format(idx, len(train_dataloader), one_loss, args.lr))

        # scheduler.step()

        one_epoch_loss /= sampler_num
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
            if args.train_image_encoder_lora:
                torch.save(model_without_ddp.sam.image_encoder.blocks.state_dict(),
                           os.path.join(output_path, 'image_encoder_blocks_' + text_model_name + '_last.pth'))
            if args.mask_word_memory:
                torch.save(model_without_ddp.memory_key.state_dict(),
                           os.path.join(output_path, 'memory_key_' + text_model_name + '_last.pth'))
                torch.save(model_without_ddp.memory_value.state_dict(),
                           os.path.join(output_path, 'memory_value_' + text_model_name + '_last.pth'))
            if args.word_memory:
                torch.save(model_without_ddp.memory_key.state_dict(),
                           os.path.join(output_path, 'memory_key_' + text_model_name + '_last.pth'))
                torch.save(model_without_ddp.memory_value.state_dict(),
                           os.path.join(output_path, 'memory_value_' + text_model_name + '_last.pth'))
            if args.mask_memory:
                torch.save(model_without_ddp.memory_key.state_dict(),
                           os.path.join(output_path, 'memory_key_' + text_model_name + '_last.pth'))
                torch.save(model_without_ddp.memory_value.state_dict(),
                           os.path.join(output_path, 'memory_value_' + text_model_name + '_last.pth'))
                torch.save(model_without_ddp.mask_conv.state_dict(),
                           os.path.join(output_path, 'mask_conv_' + text_model_name + '_last.pth'))
            if args.track_query_attn:
                torch.save(model_without_ddp.temporal_aggregation_network.state_dict(),
                           os.path.join(output_path, 'track_net_' + text_model_name + '_last.pth'))

            # each N epoch save one model
            if train_idx % 1 == 0:
                torch.save(model_without_ddp.resizer.state_dict(), os.path.join(output_path, 'resizer_' + text_model_name + "_" + str(train_idx) + '.pth'))
                if args.train_decoder:
                    torch.save(model_without_ddp.sam.mask_decoder.state_dict(), os.path.join(output_path, 'mask_decoder_' + text_model_name + "_" + str(train_idx) + '.pth'))
                if args.spatial_dynamic_fusion:
                    torch.save(model_without_ddp.dense_conv.state_dict(), os.path.join(output_path, 'dense_conv_' + text_model_name + "_" + str(train_idx) + '.pth'))
                if args.train_image_encoder_lora:
                    torch.save(model_without_ddp.sam.image_encoder.blocks.state_dict(),
                               os.path.join(output_path, 'image_encoder_blocks_' + text_model_name + "_" + str(train_idx) + '.pth'))
                if args.mask_word_memory:
                    torch.save(model_without_ddp.memory_key.state_dict(),
                            os.path.join(output_path, 'memory_key_' + text_model_name + "_" + str(train_idx) + '.pth'))
                    torch.save(model_without_ddp.memory_value.state_dict(),
                            os.path.join(output_path, 'memory_value_' + text_model_name + "_" + str(train_idx) + '.pth'))
                if args.mask_memory:
                    torch.save(model_without_ddp.memory_key.state_dict(),
                            os.path.join(output_path, 'memory_key_' + text_model_name + "_" + str(train_idx) + '.pth'))
                    torch.save(model_without_ddp.memory_value.state_dict(),
                            os.path.join(output_path, 'memory_value_' + text_model_name + "_" + str(train_idx) + '.pth'))
                    torch.save(model_without_ddp.mask_conv.state_dict(),
                            os.path.join(output_path, 'mask_conv_' + text_model_name + "_" + str(train_idx) + '.pth'))
                if args.word_memory:
                    torch.save(model_without_ddp.memory_key.state_dict(),
                            os.path.join(output_path, 'memory_key_' + text_model_name + "_" + str(train_idx) + '.pth'))
                    torch.save(model_without_ddp.memory_value.state_dict(),
                            os.path.join(output_path, 'memory_value_' + text_model_name + "_" + str(train_idx) + '.pth'))
                if args.track_query_attn:
                    torch.save(model_without_ddp.temporal_aggregation_network.state_dict(),
                               os.path.join(output_path, 'track_net_' + text_model_name + "_" + str(train_idx) + '.pth'))

    if not args.distributed or dist.get_rank() == 0:
        torch.save(model_without_ddp.resizer.state_dict(),
                   os.path.join(output_path, 'resizer_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
        if args.train_decoder:
            torch.save(model_without_ddp.sam.mask_decoder.state_dict(),
                       os.path.join(output_path, 'mask_decoder_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
        if args.spatial_dynamic_fusion:
            torch.save(model_without_ddp.dense_conv.state_dict(),
                       os.path.join(output_path, "dense_conv_" + text_model_name + "_" + str(args.train_epoch) + '.pth'))
        if args.train_image_encoder_lora:
            torch.save(model_without_ddp.sam.image_encoder.blocks.state_dict(),
                       os.path.join(output_path, 'image_encoder_blocks_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
        if args.mask_word_memory:
            torch.save(model_without_ddp.memory_key.state_dict(),
                    os.path.join(output_path, 'memory_key_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
            torch.save(model_without_ddp.memory_value.state_dict(),
                    os.path.join(output_path, 'memory_value_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
        if args.mask_memory:
            torch.save(model_without_ddp.memory_key.state_dict(),
                    os.path.join(output_path, 'memory_key_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
            torch.save(model_without_ddp.memory_value.state_dict(),
                    os.path.join(output_path, 'memory_value_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
            torch.save(model_without_ddp.mask_conv.state_dict(),
                    os.path.join(output_path, 'mask_conv_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
        if args.word_memory:
            torch.save(model_without_ddp.memory_key.state_dict(),
                    os.path.join(output_path, 'memory_key_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
            torch.save(model_without_ddp.memory_value.state_dict(),
                    os.path.join(output_path, 'memory_value_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))
        if args.track_query_attn:
            torch.save(model_without_ddp.temporal_aggregation_network.state_dict(),
                       os.path.join(output_path, 'track_net_' + text_model_name + "_" + str(args.train_epoch) + '.pth'))


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
