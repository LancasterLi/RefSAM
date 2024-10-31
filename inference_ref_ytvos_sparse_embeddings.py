import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import argparse
import warnings
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import json
import logging
import cv2
import torchvision.transforms as T
from torch.nn import functional as F

import opts
import refersam
import loss

import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
warnings.filterwarnings('ignore')

# from tools.colormap import colormap
from einops import rearrange

def main():
    # init opts
    args = opts.get_arguments()
    # args.data = './ref-davis/valid'
    print("Args:", args)

    args.masks = True
    args.batch_size == 1
    print("Inference only supports for batch size = 1")

    # create output path
    # split = args.split
    split = "valid"
    output_path = './outputs/' + args.outdir + '/' + split
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # load_dir = './outputs/' + args.load_dir

    save_visualize_path_prefix = os.path.join(output_path, split + '_images')
    if args.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    # init logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=os.path.join(output_path, 'infer_log.txt'),
        filemode='w',
    )

    logger = logging.getLogger()
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    # get palette
    palette_img = os.path.join("ref-davis/valid/Annotations/cows/00000.png")
    palette = Image.open(palette_img).getpalette()

    # assert device
    device_count = torch.cuda.device_count()
    assert device_count == 1, "inference only use 1 gpu!"
    logger.info("device_count is {}".format(device_count))

    # load model
    logger.info("======> load model")
    text_model_name = args.text_encoder
    model = refersam.Model(args, text_model_name, logger).to('cuda')

    # load checkpoint
    if args.pretrain:
        if args.proj_mlp:
            model.resizer.load_state_dict(torch.load(args.pretrain_mlp))
        if args.dense_embeddings:
            model.dense_conv.load_state_dict(torch.load(args.pretrain_dense_conv))
        if args.train_decoder:
            model.sam.mask_decoder.load_state_dict(torch.load(args.pretrain_decoder))
        if args.train_image_encoder_lora:
            model.sam.image_encoder.blocks.load_state_dict(torch.load(args.pretrain_lora_blocks))

    model.eval()

    # load data
    logger.info("load ref-ytvos valid data")
    root = './ref-youtube-vos'
    img_folder = os.path.join(root, split, "JPEGImages")
    meta_file = "./ref-youtube-vos/meta_expressions/valid/meta_expressions.json"
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    valid_test_videos = set(data.keys())

    # for some reasons the competition's validation expressions dict contains both the validation (202) &
    # test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
    # the validation expressions dict:
    test_meta_file = "./ref-youtube-vos/meta_expressions/test/meta_expressions.json"
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])
    assert len(video_list) == 202, 'error: incorrect number of validation videos'

    # inference
    logger.info('Start inference')
    to_pil = T.ToPILImage()
    # 1. for each video
    for video in tqdm(video_list):

        metas = []

        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i] # start from 0
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        # 2. For each expression
        for i in range(num_expressions):
            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            exp = " ".join(exp.lower().split()) ###
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]

            video_len = len(frames)
            # store images
            imgs = []
            for t in range(video_len):
                frame = frames[t]

                # load current image
                cur_image_path = os.path.join(img_folder, video_name, frame + ".jpg")
                cur_image = cv2.imread(cur_image_path)
                cur_image = cv2.cvtColor(cur_image, cv2.COLOR_BGR2RGB)

                origin_w, origin_h = cur_image.shape[0], cur_image.shape[1]

                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(360),
                ])

                img = transform(cur_image)
                img = img.permute(1, 2, 0)
                img = img.unsqueeze(0)

                target = {
                    'caption': exp,
                    'img': img,
                }

                with torch.no_grad():
                    output = model(img, [exp], target)

                predict_mask = output[0][0] # [1, 480, 910]

                new_predict_mask = F.interpolate(predict_mask.unsqueeze(0), size=(origin_w, origin_h), mode='bilinear', align_corners=False)

                # save masks
                bool_tensor_mask = new_predict_mask[0] > 0.01

                float_tensor_mask = bool_tensor_mask.float().squeeze(0) # [1 720 1280] -> [720 1280]
                array_mask = float_tensor_mask.detach().cpu().numpy()

                mask = to_pil(array_mask * 255).convert('L') # 1280 * 720

                # save binary image
                save_path = os.path.join(output_path, video_name, exp_id)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                frame_name = frames[t]

                save_file = os.path.join(save_path, frame_name + ".png")
                mask.save(save_file)

                if args.visualize:
                    # original
                    img_path = os.path.join(img_folder, video_name, frame + '.jpg')
                    source_img = Image.open(img_path) # PIL image

                    # draw = ImageDraw.Draw(source_img)
                    
                    # draw mask
                    result = Image.new('RGBA', source_img.size)
                    # result = Image.composite(source_img, result, mask)
                    result = Image.blend(source_img, mask, alpha=0.5)

                    # save
                    save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video, str(i))
                    if not os.path.exists(save_visualize_path_dir):
                        os.makedirs(save_visualize_path_dir)
                    save_visualize_path = os.path.join(save_visualize_path_dir, frame + '.png')
                    result.save(save_visualize_path)


if __name__ == "__main__":
    main()