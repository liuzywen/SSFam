import os
import random

from PIL import Image, ImageEnhance
import torch.utils.data as data
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def get_prompt(x, y, points_num, imgsize):
    point_list = []
    label_list = []
    t = 0
    while len(point_list) < points_num and t < 10000:
        t = t + 1
        random_x = np.random.randint(0, imgsize)
        random_y = np.random.randint(0, imgsize)
        x_value = x.getpixel((random_x, random_y))
        y_value = y.getpixel((random_x, random_y))
        # 判断前景或者背景
        if x_value == 255 and y_value == 255:
            point_list.append([random_x, random_y])
            label_list.append(1)
        elif x_value == 0 and y_value == 255:
            point_list.append([random_x, random_y])
            label_list.append(0)

    if len(point_list) < 1:
        point_list.append([0, 0])
        label_list.append(-1)

    # 将随机选择的点坐标转换为tensor张量作为输入点
    input_point = torch.tensor(point_list)
    # 定义对应输入点的标签
    input_label = torch.tensor(label_list)
    return input_point, input_label


class SalObjDataset(data.Dataset):
    def __init__(
            self, image_root, depth_root, gt_root, mask_root, gray_root, trainsize
    ):
        self.trainsize = trainsize
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [os.path.join(depth_root, f) for f in os.listdir(depth_root) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.masks = [os.path.join(mask_root, f) for f in os.listdir(mask_root) if
                      f.endswith('.jpg') or f.endswith('.png')]
        self.grays = [os.path.join(gray_root, f) for f in os.listdir(gray_root) if
                      f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        self.masks = sorted(self.masks)
        self.grays = sorted(self.grays)
        self.size = len(self.images)

        self.resize_transform = transforms.Resize((self.trainsize, self.trainsize))
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.up = transforms.Resize((self.trainsize, self.trainsize),
                                    interpolation=transforms.InterpolationMode.NEAREST)
        self.points_num = 10

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        depth = self.rgb_loader(self.depths[index])
        gt = self.binary_loader(self.gts[index])
        mask = self.binary_loader(self.masks[index])
        gray = self.binary_loader(self.grays[index])

        gt_map = self.up(gt)
        mask_map = self.up(mask)
        input_point, input_label = get_prompt(gt_map, mask_map, self.points_num, self.trainsize)

        image = self.normalize_transform(self.to_tensor_transform(self.resize_transform(image)))
        depth = self.to_tensor_transform(self.resize_transform(depth))
        gt = self.to_tensor_transform(self.resize_transform(gt))
        mask = self.to_tensor_transform(self.resize_transform(mask))
        gray = self.to_tensor_transform(self.resize_transform(gray))

        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        return image, depth, gt, mask, gray, input_point, input_label, name
        # return image, depth, gt, mask, gray

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader(
        image_root, depth_root, gt_root, mask_root, gray_root, batchsize, trainsize,
        shuffle=True, num_workers=0, pin_memory=True
):
    dataset = SalObjDataset(
        image_root, depth_root, gt_root, mask_root, gray_root, trainsize,
    )
    data_loader = data.DataLoader(
        dataset=dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory
    )
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth = self.rgb_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
            # name = name.split('.jpg')[0] + '.jpg'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
