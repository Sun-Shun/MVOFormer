import sys
sys.path.append('core')
import argparse
import os
import cv2
import math
import numpy as np
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from config.parser import parse_args

import core.datasets
from core.raft import RAFT
from core.utils.flow_viz import flow_to_image
from core.utils.utils import load_ckpt
from joblib import Parallel, delayed
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])

def vis_heatmap(name, image, heatmap):
    # theta = 0.01
    # print(heatmap.max(), heatmap.min(), heatmap.mean())
    heatmap = heatmap[:, :, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap = heatmap > 0.01
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = image * 0.3 + colored_heatmap * 0.7
    # Create a color bar
    height, width = image.shape[:2]
    color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)  # Adjust the height and colormap as needed
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    combined_image = add_color_bar_to_image(overlay, color_bar, 'vertical')
    cv2.imwrite(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def viz(flo, flo_vis, scene, imfile):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    img_name = imfile.split(".")[0]
    root_path = scene.replace('test_img', 'test_flow_sea')
    vis_path = scene.replace('test_img', 'test_flow_vis')
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    # if not os.path.exists(vis_path):
    #     os.makedirs(vis_path)
    flo_save_path = os.path.join(root_path, img_name + ".npy")

    img_path = os.path.join(vis_path, img_name+".png")
    # np.save(img_path, flo)  # 保存文件
    # cv2.imwrite(img_path, flo_vis)

def viz_c(feature, scene, imfile):
    # feature = feature[0].permute(1, 2, 0).cpu().numpy()
    feature = feature[0].to(dtype=torch.float16).cpu().numpy()
    img_name = imfile.split(".")[0]
    root_path = scene.replace('test_img', 'test_context')
    root_path = root_path.replace('数据', 'My PSSD')
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    feature_save_path = os.path.join(root_path, img_name + ".npy")
    np.save(feature_save_path, feature)  # 保存文件

def viz_info(info, scene, imfile):
    # feature = feature[0].permute(1, 2, 0).cpu().numpy()
    info = info[0].to(dtype=torch.float16).cpu().numpy()
    img_name = imfile.split(".")[0]
    root_path = scene.replace('test_img', 'test_info')
    root_path = root_path.replace('数据', 'My PSSD')
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    feature_save_path = os.path.join(root_path, img_name + ".npy")
    np.save(feature_save_path, info)  # 保存文件

def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    cnet_final = output['cnet']

    return flow_final, info_final, cnet_final

def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info, cnet = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down, cnet

@torch.no_grad()
def demo_data(path, args, model, image1, image2,imfile1):

    H, W = image1.shape[2:]
    flow, info, cnet= calc_flow(args, model, image1, image2)

    # combined = torch.cat([info, cnet_upsampled], dim=1)
    # flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    viz_c(cnet, path, imfile1)
    viz_info(info, path, imfile1)
    # heatmap = get_heatmap(info, args)
    # vis_heatmap(f"{path}heatmap.jpg", image1[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())

@torch.no_grad()
def demo_custom(model, args, device, temp_path):
    namee_images = os.listdir(temp_path)
    namee_images.sort()
    if len(namee_images) < 1:
        print("empty")

    for i, (imfile1, imfile2) in enumerate(zip(namee_images[:-1], namee_images[1:])):
        image1 = cv2.imread(temp_path + '/' + imfile1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

        image2 = cv2.imread(temp_path + '/' + imfile2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
        # image1 = F.interpolate(image1.unsqueeze(0), scale_factor=1/4,
        #                                mode='bilinear', align_corners=False)
        # image1 = image1.squeeze(0)

        image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
        # image2 = F.interpolate(image2.unsqueeze(0), scale_factor=1/4,
        #                                mode='bilinear', align_corners=False)
        # image2 = image2.squeeze(0)

        H, W = image1.shape[1:]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)
        demo_data(temp_path, args, model, image1, image2, imfile1)

    print(temp_path)
    print("-----------end-------------")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--path', help='checkpoint path', type=str, default=None)
    parser.add_argument('--url', help='checkpoint url', type=str, default=None)
    parser.add_argument('--device', help='inference device', type=str, default='cuda')
    args = parse_args(parser)
    if args.path is None and args.url is None:
        raise ValueError("Either --path or --url must be provided")
    if args.path is not None:
        model = RAFT(args)
        load_ckpt(model, args.path)
    else:
        model = RAFT.from_pretrained(args.url, args=args)
        
    if args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    train_flow_list = []
    train_scene = '/media/ssw/数据/tartanair/train'
    test_scene = '/media/ssw/数据/tartanair/test'

    train_scene = glob('/media/ssw/数据/tartanair/test/test_img' + '/*')
    train_scene.sort()

    train_flow_list = train_scene
    results = Parallel(n_jobs=1)(delayed(demo_custom)(model, args, device, patha) for patha in train_flow_list)

    # demo_custom(model, args, device=device)

if __name__ == '__main__':
    main()