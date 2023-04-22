import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
import torchvision.transforms.functional as F
from PIL import Image
import seaborn as sns

from dataset import MaskBaseDataset, MaskPreprocessDataset# dataset.py
from dataset import TestDataset
from loss import create_criterion # loss.py
from f1score import get_F1_Score # f1score.py
from submission import submission # submission.py
from inference import inference # inference.py
import wandb

from dataset import MaskBaseDataset # dataset.py
from dataset import TestDataset
from loss import create_criterion # loss.py
from f1score import get_F1_Score # f1score.py
from submission import submission # submission.py
from inference import inference # inference.py
import wandb
import shutil

class RandomGaussianBlur(object):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        
    def __call__(self, img):
        if np.random.rand() < 0.5:
            return img
        else:
            return F.gaussian_blur(img, kernel_size=self.kernel_size, sigma=(0.1, 2.0))    


def random_transform(image):
    scale = (0.005, 0.025)
    ratio = (0.3, 3.3)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(p=0.6),
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                    transforms.RandomErasing(p=0.7, scale=scale, ratio=ratio),
                                    transforms.RandomErasing(p=0.5, scale=scale, ratio=ratio),
                                    RandomGaussianBlur(kernel_size=3),
                                    transforms.RandomRotation(5),
                                    transforms.ToPILImage()
                                   ])
    return transform(image)    


def AddAugmentation(label_paths, idx, aug_size, aug_dir_name):
    aug_dir_name = aug_dir_name
    idx2label = ['mask']*6+['incorrect']*6+['normal']*6
    os.makedirs(aug_dir_name, exist_ok = True)
    idx = int(idx)
    aug_size = int(aug_size)
    data_size = len(label_paths[idx])
    repeat = aug_size - data_size

    print("입력 : ", aug_size,  "실제 데이터 :", data_size, "차이 : ",repeat)
    if repeat < 0: #마이너스면 삭제 처리해줌 
        print("삭제진행")
        for _ in tqdm(range(-repeat)):        
            if len(label_paths[idx]) <= 0:  # 이미지 경로가 남아있지 않으면 중지
                break
            random_int = random.randint(0, len(label_paths[idx])-1)
            img_path = label_paths[idx][random_int]
            os.remove(img_path)
            label_paths[idx].remove(img_path)
        print("삭제완료")

    else : #플러스면 증강 처리해줌
        print("증강진행")
        for _ in tqdm(range(repeat)):
            random_int = random.randint(0,data_size-1)
            img_id = label_paths[idx][random_int].split('/')[-2]
            img = Image.open(label_paths[idx][random_int])
            img = random_transform(img)
            img.save(os.path.join(aug_dir_name, img_id, idx2label[idx]+str(_+10)+'.jpg'))
        print("증강완료")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='MaskPreprocessDataset', help='dataset augmentation type (default: MaskPreprocessDataset)')
#     parser.add_argument('--delplus', type=int, default=0,choices=[1, 0], help = 'want? (y : 1 enter ,n : 0 enter 1를 입력하면 지정 텍스트 파일을 읽어 실행됨)') # 무조건 실행되므로 필요없음
    parser.add_argument('--aug_dir_name', type=str, default='/opt/ml/input/data/augmentation_delete_data', help = 'create preprocess dataset folder')
    
    if os.path.exists('/opt/ml/input/augmentation_delete_data'):
        print('augmentation_delete_data is already exists')
        exit()

    args = parser.parse_args()
    print(args)
    
    # 원본 데이터
    src_dir = '/opt/ml/input/data' 
    aug_dir_name = args.aug_dir_name+'/train/images'
    # 증강 및 삭제할 데이터
    dst_dir = aug_dir_name

    def copy_data(src, dst):
        
        shutil.copytree(src, dst)

    src, dst = '/opt/ml/input/data', '/opt/ml/input/augmentation_delete_data'
    copy_data(src,dst)
    
#     delplus = args.delplus
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskPreprocessDataset
    dataset = dataset_module(
        data_dir=aug_dir_name,
        outlier_remove=False
    )
    
    # -- delplus 다현 추가 부분
    with open('./delplustxt.txt', 'r') as f:
        for line in f:
            idx,size = line.strip().split(',')
            AddAugmentation(dataset.label_paths, idx, size, aug_dir_name)
    print('datapreprocess is done! if you want to use preprocessed data, put data_dir parser --data_dir /opt/ml/input/augmentation_delete_data')
            
# python datapreprocess.py --aug_dir_name /opt/ml/input/augmentation_delete_data