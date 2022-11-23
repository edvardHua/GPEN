# -*- coding: utf-8 -*-
# @Time : 2022/11/21 17:19
# @Author : zihua.zeng
# @File : test_dgf.py
#
# 测试 DGF 上采样的效果
#

import os.path

import cv2
import torch
import numpy as np

from torchvision import transforms
from guided_filter_pytorch.guided_filter import FastGuidedFilter

from torchprofile import profile_macs
from pprint import pprint


def prepare():
    img = cv2.imread(
        "/Users/zihua.zeng/Workspace/demo-py/image_processing/GPEN/examples/out_dgf/Anoy=8MA-ByMWgAhhQEPAEY_137800_COMP.jpg")
    h, w, _ = img.shape
    ih = img[:, :w // 2, :]
    oh = img[:, w // 2:, :]
    cv2.imwrite("/Users/zihua.zeng/Workspace/demo-py/image_processing/GPEN/examples/out_dgf/ih/1.jpg", ih)
    cv2.imwrite("/Users/zihua.zeng/Workspace/demo-py/image_processing/GPEN/examples/out_dgf/oh/1.jpg", oh)


def test_dgf():
    base_path = "/Users/zihua.zeng/Workspace/demo-py/image_processing/GPEN/examples/out_dgf/"
    il = cv2.imread(os.path.join(base_path, "il/1.jpg"))
    ol = cv2.imread(os.path.join(base_path, "ol/1.jpg"))
    ol_dup = ol.copy()
    ih = cv2.imread(os.path.join(base_path, "ih/1.jpg"))
    ih_dup = ih.copy()
    hh, hw, _ = ih.shape

    filter = FastGuidedFilter(3)

    trans = transforms.Compose([transforms.ToTensor()])
    il = torch.unsqueeze(trans(il), 0)
    ol = torch.unsqueeze(trans(ol), 0)
    ih = torch.unsqueeze(trans(ih), 0)
    hr_y = filter(il, ol, ih)

    macs = profile_macs(filter, (il, ol, ih))
    pprint(macs / 1e6)

    hr_y = hr_y.numpy().squeeze()
    hr_y = np.transpose(hr_y, (1, 2, 0))
    hr_y = (hr_y * 255.).clip(0, 255).astype(np.uint8)

    # hr_y = np.hstack([ih_dup, cv2.resize(ol_dup, (hw, hh)), hr_y, cv2.imread(os.path.join(base_path, "oh/1.jpg"))])
    hr_y = np.hstack([ih_dup, hr_y, cv2.imread(os.path.join(base_path, "oh/1.jpg"))])
    cv2.imwrite("test.jpg", hr_y)


if __name__ == '__main__':
    test_dgf()
    # prepare()
