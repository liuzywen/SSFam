import torch
import torch.nn.functional as F
import sys

sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from data import test_dataset
from SAM_Model.model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--img_size', type=int, default=1024, help='training dataset size')
parser.add_argument('--data_dir', default=' ', help='test dataset path')
parser.add_argument('--model_type', type=str, default='vit_l', help='weight for edge loss')
parser.add_argument('--checkpoint', type=str,  default=' ', help='test from checkpoints')
parser.add_argument('--save_dir', default='./test_maps/', help='path where to save predicted maps')
opt = parser.parse_args()

model = Model(opt)
model.load_state_dict(torch.load(opt.checkpoint))
for param in model.parameters():
    param.requires_grad_(False)
model.cuda()
model.eval()

# test
test_datasets = ['DES', 'LFSD', 'NJU2K_Test', 'NLPR_Test', 'SIP', 'STERE', 'SSD']
for dataset in test_datasets:
    save_path = os.path.join(opt.save_dir, dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_loader = test_dataset(
        os.path.join(opt.data_dir, 'test_data', 'img', dataset) + '/',
        os.path.join(opt.data_dir, 'test_data', 'gt', dataset) + '/',
        os.path.join(opt.data_dir, 'test_data', 'depth', dataset) + '/', opt.img_size)

    for i in range(test_loader.size):
        image, gt, depth, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()

        res, _ = model(image, depth, None)

        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ', save_path + name)
        res = np.round(res * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_path, name), res)
    print('Test Done!')
