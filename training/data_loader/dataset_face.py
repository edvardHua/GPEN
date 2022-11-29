import numpy as np
import cv2
import os
import glob
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import degradations


class GFPGAN_degradation(object):
    """
    """

    def __init__(self):
        self.kernel_list = ['iso', 'aniso']
        self.kernel_prob = [0.5, 0.5]
        self.blur_kernel_size = 41
        self.blur_sigma = [0.1, 10]
        self.downsample_range = [0.8, 8]
        self.noise_range = [0, 20]
        self.jpeg_range = [60, 100]
        self.gray_prob = 0.2
        self.color_jitter_prob = 0.0
        self.color_jitter_pt_prob = 0.0
        self.shift = 20 / 255.

    def degrade_process(self, img_gt):
        if random.random() > 0.5:
            img_gt = cv2.flip(img_gt, 1)

        h, w = img_gt.shape[:2]

        # random color jitter 
        if np.random.uniform() < self.color_jitter_prob:
            jitter_val = np.random.uniform(-self.shift, self.shift, 3).astype(np.float32)
            img_gt = img_gt + jitter_val
            img_gt = np.clip(img_gt, 0, 1)

        # random grayscale
        if np.random.uniform() < self.gray_prob:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])

        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)

        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # round and clip
        img_lq = np.clip((img_lq * 255.0).round(), 0, 255) / 255.

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        return img_gt, img_lq


class Shopee_degradation():
    """
    适合线上场景的数据退化逻辑
    去除 color jitter, flip, grayscale
    调整一些退化参数的取值
    输入图像为 512x512
    """

    def __init__(self):
        self.kernel_list = ['iso', 'aniso']
        self.kernel_prob = [0.5, 0.5]
        self.blur_kernel_size = 27
        self.blur_sigma = [0.1, 4]
        self.downsample_range = [2, 6]
        self.noise_range = [0, 12]
        self.jpeg_range = [60, 100]
        self.gray_prob = 0.2
        self.shift = 20 / 255.

    def degrade_process(self, img_gt):
        img_gt_copy = img_gt.copy()
        h, w = img_gt.shape[:2]

        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)

        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)

        if self.noise_range:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)

        if self.jpeg_range:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        return img_gt_copy, img_lq


class FaceDataset(Dataset):
    def __init__(self, path, resolution=512, split_horizon=False):
        self.resolution = resolution

        self.HQ_imgs = glob.glob(os.path.join(path, '*.*'))
        self.length = len(self.HQ_imgs)

        # self.degrader = GFPGAN_degradation()
        self.degrader = Shopee_degradation()
        self.split_horizon = split_horizon

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_gt = cv2.imread(self.HQ_imgs[index], cv2.IMREAD_COLOR)
        if self.split_horizon:
            h, w, _ = img_gt.shape
            img_gt = img_gt[:, w // 2:, :]

        img_gt = cv2.resize(img_gt, (self.resolution, self.resolution), interpolation=cv2.INTER_AREA)

        # BFR degradation
        # We adopt the degradation of GFPGAN for simplicity, which however differs from our implementation in the paper.
        # Data degradation plays a key role in BFR. Please replace it with your own methods.
        img_gt = img_gt.astype(np.float32) / 255.
        img_gt, img_lq = self.degrader.degrade_process(img_gt)

        img_gt = (torch.from_numpy(img_gt) - 0.5) / 0.5
        img_lq = (torch.from_numpy(img_lq) - 0.5) / 0.5

        img_gt = img_gt.permute(2, 0, 1).flip(0)  # BGR->RGB
        img_lq = img_lq.permute(2, 0, 1).flip(0)  # BGR->RGB

        return img_lq, img_gt


def test_dataset():
    from torch.utils import data
    from porsche.utils.torch_knife import tensor2img

    test_path = "/Users/zihua.zeng/Dataset/人像增强/人脸增强/face_enhencer_1123/val"
    dataset = FaceDataset(test_path, 256, True)
    loader = data.DataLoader(dataset,
                             batch_size=1,
                             drop_last=True)

    for lq, gt in loader:
        lq = tensor2img(lq, 0.5, 0.5)
        gt = tensor2img(gt, 0.5, 0.5)
        print(lq[0].shape, gt[0].shape)
        vs = np.hstack([lq[0], gt[0]])
        cv2.imshow("test", vs)
        cv2.waitKey(0)
        break


def test_degradation():
    path = "/Users/zihua.zeng/Dataset/人像增强/人脸增强/face_enhencer_1123/val/AnoyvJIAWGObWQAhdgEHAEY_232050-0.jpg"
    face_vs_image = cv2.imread(path)
    h, w, _ = face_vs_image.shape
    gt_lface = face_vs_image[:, :w // 2, :]
    gt_face = face_vs_image[:, w // 2:, :]
    degrader = Shopee_degradation()
    gt_face = gt_face.astype(np.float32) / 255.

    out_path = "dummy_image"
    os.makedirs(out_path, exist_ok=True)

    for _ in range(300):
        img_gt, img_lq = degrader.degrade_process(gt_face)
        img_lq = (img_lq * 255.).astype(np.uint8)
        img_gt = (img_gt * 255.).astype(np.uint8)
        union = np.hstack([gt_lface, img_lq, img_gt])
        cv2.imwrite(os.path.join(out_path, "%d.jpg" % _), union)
        print(_)


if __name__ == '__main__':
    test_degradation()
    pass
