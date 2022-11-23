# -*- coding: utf-8 -*-
# @Time : 2022/11/22 11:46
# @Author : zihua.zeng
# @File : gen_face_dataset.py
#
# 使用 GPEN 生成数据集来训练小的 GAN 模型
# 1. 先对人脸 align
# 2. 然后用预训练的 gpen 跑一下生成 paired 数据集

import os
import cv2
import argparse

import numpy as np

from face_enhancement import FaceEnhancement
from tqdm import tqdm


def param():
    ns = argparse.Namespace(model='GPEN-BFR-512', task='FaceEnhancement', key=None, in_size=512, out_size=None,
                            channel_multiplier=2, narrow=1.0, alpha=1, use_sr=True, use_cuda=False, save_face=True,
                            aligned=False, sr_model='realesrnet', sr_scale=2, tile_size=0,
                            indir='/Users/zihua.zeng/Workspace/imageadaptive3dlut/shopee_videos/mild_blur_crop_faces',
                            outdir='examples/lq_hq_faces',
                            ext='.jpg')

    return ns


def gen():
    args = param()
    processer = FaceEnhancement(args, in_size=args.in_size, model=args.model, use_sr=args.use_sr,
                                device='cuda' if args.use_cuda else 'cpu')
    os.makedirs(args.outdir, exist_ok=True)

    for fn in tqdm(os.listdir(args.indir)):
        if not fn.endswith(".jpg"):
            continue

        img = cv2.imread(os.path.join(args.indir, fn))
        img_out, orig_faces, enhanced_faces = processer.process(img, aligned=args.aligned)

        if len(enhanced_faces) == 0:
            continue

        union = np.hstack([orig_faces[0], enhanced_faces[0]])
        cv2.imwrite(os.path.join(args.outdir, fn), union)


if __name__ == '__main__':
    gen()
