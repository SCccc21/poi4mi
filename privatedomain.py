import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as tvmodels
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import shutil

import random
from classify import *
from utils import *
# check, if file exists, make link
 
if __name__ == '__main__':
    global args

    file = "./config/classify_cifar.json"
    args = load_json(json_file=file)
    file_path = '/home/sichen/mi/GMI-code/cifar10/data/CIFAR_imgs/train.txt'
    save_img_dir = "./private_domain"
    os.makedirs(save_img_dir, exist_ok=True)

    private_set, private_loader = init_dataloader(args, file_path, batch_size=1, mode="attack", iterator=False)
    # cnt = 0
    # for i, (imgs, iden) in enumerate(private_loader):
    #     print("-------------- Process batch {} -----------------".format(i))
    #     for b in range(imgs.shape[0]):
    #         cnt += 1
    #         save_tensor_images(imgs[b], os.path.join(save_img_dir, "batch_{}_{}th_iden_{}.png".format(i, b, iden[b]+1)))
    #         if iden[b] == 299:
    #             print("Last person!")
    #         # print("save image of iden {}".format(iden[b]+1))

    # print(cnt)
    for i, (imgs, iden) in enumerate(private_loader):
        print("save {} image of iden {}".format(i, iden[0]+1))
        save_tensor_images(imgs, os.path.join(save_img_dir, "{}th_iden_{}.png".format(i, iden[0]+1)))