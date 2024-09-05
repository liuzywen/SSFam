import itertools

import cv2
import torch

from SAM_Model.model import Model
import numpy as np
import pdb, os, argparse
from datetime import datetime
from data import get_loader, test_dataset

from utils import adjust_lr
import os
import logging
import smoothness
from tools import *
from lscloss import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--img_size', type=int, default=1024, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
parser.add_argument('--sm_loss_weight', type=float, default=0.3, help='weight for smoothness loss')
parser.add_argument('--freeze_image_encoder', type=bool, default=True, help='True or False')
parser.add_argument('--freeze_prompt_encoder', type=bool, default=True, help='True or False')
parser.add_argument('--freeze_mask_decoder', type=bool, default=False, help='True or False')
parser.add_argument('--model_type', type=str, default='vit_l', help='vit_b, vit_l, vit_h')
parser.add_argument('--checkpoint', type=str, default=' ', help='train from checkpoints')
parser.add_argument('--save_path', type=str, default=' ', help='the path to save models and logs')
parser.add_argument('--val_RGBD', default='DES', help='val dataset')
parser.add_argument('--val_RGBD_dir', default=' ', help='path where to save testing data')
parser.add_argument('--train_RGBD_dir', default=' ', help='path where to save training data')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))
model = Model(opt)
model.setup()
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), opt.lr)

train_loader = get_loader(os.path.join(opt.train_RGBD_dir, 'train_data', 'img') + '/',
                          os.path.join(opt.train_RGBD_dir, 'train_data', 'depth') + '/',
                          os.path.join(opt.train_RGBD_dir, 'train_data', 'gt') + '/',
                          os.path.join(opt.train_RGBD_dir, 'train_data', 'mask') + '/',
                          os.path.join(opt.train_RGBD_dir, 'train_data', 'gray') + '/',
                          batchsize=opt.batchsize, trainsize=opt.img_size)

test_loader = test_dataset(
    os.path.join(opt.val_RGBD_dir, 'test_data', 'img', opt.val_RGBD) + '/',
    os.path.join(opt.val_RGBD_dir, 'test_data', 'gt', opt.val_RGBD) + '/',
    os.path.join(opt.val_RGBD_dir, 'test_data', 'depth', opt.val_RGBD) + '/', opt.img_size)


total_step = len(train_loader)

CE = torch.nn.BCELoss()
smooth_loss = smoothness.smoothness_loss(size_average=True)

best_mae = 1
best_epoch = 0


loss_lsc = LocalSaliencyCoherence().cuda()
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
save_path = opt.save_path

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("scribbleNet-Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};img_size:{};decay_rate:{};save_path:{};decay_epoch:{};'
    'freeze_image_encoder:{};freeze_prompt_encoder:{};freeze_mask_decoder:{}'.
        format(opt.epoch, opt.lr, opt.batchsize, opt.img_size, opt.decay_rate, save_path, opt.decay_epoch,
               opt.freeze_image_encoder, opt.freeze_prompt_encoder, opt.freeze_mask_decoder))


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        images, depths, gts, masks, grays, input_point, input_label, name = pack
        images = images.cuda()
        depths = depths.cuda()
        gts = gts.cuda()
        masks = masks.cuda()
        grays = grays.cuda()
        input_point = input_point.cuda()
        input_label = input_label.cuda()

        point_prompts = (input_point, input_label)

        SAM_mask1, SAM_mask2 = model(images, depths, point_prompts)

        str_loss = structure_loss(SAM_mask1, torch.sigmoid(SAM_mask2.detach()))

        SAM_mask_prob2 = torch.sigmoid(SAM_mask2)

        img_size = images.size(2) * images.size(3) * images.size(0)
        ratio = img_size / (torch.sum(masks) + 1e-8)
        sal_loss2 = ratio * CE(SAM_mask_prob2 * masks, gts * masks)
        sm_loss2 = opt.sm_loss_weight * smooth_loss(SAM_mask_prob2, grays)

        images_ = F.interpolate(images, scale_factor=0.25, mode="bilinear", align_corners=True)
        sample_rgb = {'rgb': images_}
        result_final_ = F.interpolate(SAM_mask_prob2, scale_factor=0.25, mode="bilinear", align_corners=True)
        lsc_loss2 = loss_lsc(result_final_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample_rgb,
                             images_.shape[2], images_.shape[3])['loss']

        loss = (str_loss + sal_loss2 + sm_loss2 + lsc_loss2).mean()

        loss.backward()
        optimizer.step()

        if i % 100 == 0 or i == total_step or i == 1:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], str_loss: {:.4f}, '
                  'sal_loss2: {:.4f}, smooth_loss2: {:.4f}, total_loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, str_loss.data,
                         sal_loss2.data, sm_loss2.data, loss.data))
            logging.info(
                '#TRAIN#:{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], str_loss: {:.4f}, '
                'sal_loss2: {:.4f}, smooth_loss2: {:.4f}, total_loss: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, str_loss.data,
                           sal_loss2.data, sm_loss2.data, loss.data))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 500 == 0:
        torch.save(model.state_dict(), save_path + 'H-Adapter' + '_%d' % epoch + '.pth')


def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            # print(i)
            image, gt, depth, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            res, _ = model(image, depth, None)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
            best_epoch = epoch
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(),
                           save_path + '' + str(best_mae) + 'best' + '_%d' % epoch + '.pth')
                print('best epoch:{}'.format(epoch))

        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{} '.
                     format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Starting!")
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
        test(test_loader, model, epoch, save_path)
