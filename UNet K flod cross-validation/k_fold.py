import os
from PIL import Image
import torch
import torchvision
import sys
from torchvision.models import densenet161, resnet50, resnet101,resnet18
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from PIL import Image
from torch import optim
from torch import nn
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from time import time
import time
import  csv


def get_k_fold_data(k, k1, image_dir):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1##K折交叉验证K大于1
    file = open(image_dir, 'r', encoding='utf-8',newline="")
    reader = csv.reader(file)
    imgs_ls = []
    for line in reader:
        imgs_ls.append(line)
    #print(len(imgs_ls))
    file.close()

    avg = len(imgs_ls) // k

    f1 = open('./train_k.txt', 'w',newline='')
    f2 = open('./test_k.txt', 'w',newline='')
    writer1 = csv.writer(f1)
    writer2 = csv.writer(f2)
    for i, row in enumerate(imgs_ls):
        #print(row)
        if (i // avg) == k1:
            writer2.writerow(row)
        else:
            writer1.writerow(row)
    f1.close()
    f2.close()

