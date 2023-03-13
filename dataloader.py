import os, utils, torchvision
import json, PIL, time, random
import torch, math, cv2

import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F 
import torch.utils.data as data
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from sklearn.model_selection import train_test_split

# mnist_path = "/home/sichen/data/mnist"
# mnist_img_path = "/home/sichen/data/MNIST_imgs"
cifar_path = "./data/CIFAR"
cifar_img_path = "./data/CIFAR_imgs"
os.makedirs(cifar_path, exist_ok=True)
os.makedirs(cifar_img_path, exist_ok=True)



class MNISTSubset(datasets.MNIST):
    def __init__(self, root, train=True, download=True, **kwargs):
        super(MNISTSubset, self).__init__(root, train=train, download=download)

        self.setup_exclude()
        labels = np.array(self.targets)
        exclude = np.array(self.exclude_list).reshape(1, -1)
        mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)
        print('-----USE EXLUDE LIST {}-----'.format(self.exclude_list))

        self.data = self.data[mask]
        self.targets = labels[mask].tolist()


class MNIST01(MNISTSubset):
    def setup_exclude(self):
        self.exclude_list = [2,3,4,5,6,7,8,9]
        self.transform = transforms.Compose([
                                        transforms.Lambda(lambda x: x.convert('RGB')),
                                        transforms.Resize((32, 32)),
                                        # transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5)
                                        ])


class STLSubset(datasets.STL10):
    def __init__(self, root, split='train', download=True, **kwargs):
        super(STLSubset, self).__init__(root, split=split, download=download)

        self.setup_exclude()
        labels = np.array(self.labels)
        exclude = np.array(self.exclude_list).reshape(1, -1)
        mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)
        print('-----USE EXLUDE LIST {}-----'.format(self.exclude_list))

        self.data = self.data[mask]
        self.labels = (np.ones_like(labels)*3).tolist()

class STL_monkey(STLSubset):
    def setup_exclude(self):
        self.exclude_list = [0,1,2,3,4,5,6,8,9]
        self.transform = transforms.Compose([
                                        transforms.Resize((32, 32)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()
                                        ])


class ImageFolder(data.Dataset):
    def __init__(self, args, file_path, mode):
        self.args = args
        self.mode = mode
        self.img_path = args["dataset"]["img_path"]
        self.model_name = args["dataset"]["model_name"]
        # self.img_list = os.listdir(self.img_path)
        self.processor = self.get_processor()
        self.name_list, self.label_list = self.get_list(file_path) 
        self.image_list = self.load_img()
        self.num_img = len(self.image_list)
        self.n_classes = args["dataset"]["n_classes"]
        # print("Load " + str(self.num_img) + " images")

    
    def get_list(self, file_path):
        name_list, label_list = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            if self.mode == "gan":
                img_name = line.strip()
            else:
                img_name, iden = line.strip().split(' ')
                label_list.append(int(iden))
            name_list.append(img_name)
            

        return name_list, label_list

    
    def load_img(self):
        img_list = []
        for i, img_name in enumerate(self.name_list):
            if img_name.endswith(".png"):
                path = self.img_path + "/" + img_name
                img = PIL.Image.open(path)
                img = img.convert('RGB')
                img_list.append(img)
        return img_list
    
    
    def get_processor(self):
        if self.args['dataset']['name'] == "cifar10":
            re_size = 32
        else:
            re_size = 64
            
        crop_size = 32
        offset_height = (32 - crop_size) // 2
        offset_width = (32 - crop_size) // 2

        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        proc = []
        if self.mode == "train":
            proc.append(transforms.ToTensor())
            proc.append(transforms.Lambda(crop))
            proc.append(transforms.ToPILImage())
            proc.append(transforms.Resize((re_size, re_size)))
            proc.append(transforms.RandomHorizontalFlip(p=0.5))
            proc.append(transforms.ToTensor())
        else:
            proc.append(transforms.ToTensor())
            proc.append(transforms.Lambda(crop))
            proc.append(transforms.ToPILImage())
            proc.append(transforms.Resize((re_size, re_size)))
            proc.append(transforms.ToTensor())
        
            
        return transforms.Compose(proc)

    def __getitem__(self, index):
        processer = self.get_processor()
        # print('before', index)
        img = processer(self.image_list[index])
        # print('after', index)
        if self.mode == "gan":
            return img
        label = self.label_list[index]
        # print(label)
        # print('--------')

        return img, label

    def __len__(self):
        return self.num_img

class GrayFolder(data.Dataset):
    def __init__(self, args, file_path, mode):
        self.args = args
        self.mode = mode
        self.img_path = args["dataset"]["img_path"]
        self.img_list = os.listdir(self.img_path)
        self.processor = self.get_processor()
        self.name_list, self.label_list = self.get_list(file_path) 
        self.image_list = self.load_img()
        self.num_img = len(self.image_list)
        self.n_classes = args["dataset"]["n_classes"]
        print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path):
        name_list, label_list = [], []
        f = open(file_path, "r")
        for line in f.readlines():
            if self.mode == "gan":
                img_name = line.strip()
            else:
                img_name, iden = line.strip().split(' ')
                label_list.append(int(iden))
            name_list.append(img_name)

        return name_list, label_list

    
    def load_img(self):
        img_list = []
        for i, img_name in enumerate(self.name_list):
            if img_name.endswith(".png"):
                path = self.img_path + "/" + img_name
                img = PIL.Image.open(path)
                img = img.convert('L')
                img_list.append(img)
        return img_list
    
    def get_processor(self):
        proc = []
        if self.args['dataset']['name'] == "mnist":
            re_size = 32
        else:
            re_size = 64
        proc.append(transforms.Resize((re_size, re_size)))
        proc.append(transforms.ToTensor())
            
        return transforms.Compose(proc)

    def __getitem__(self, index):
        processer = self.get_processor()
        img = processer(self.image_list[index])
        if self.mode == "gan":
            return img
        label = self.label_list[index]
    
        return img, label

    def __len__(self):
        return self.num_img

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(mnist_path, train=True, transform=transform, download=True)
    testset = torchvision.datasets.MNIST(mnist_path, train=False, transform=transform, download=True)

    train_loader = DataLoader(trainset, batch_size=1)
    test_loader = DataLoader(testset, batch_size=1)
    cnt = 0

    for imgs, labels in train_loader:
        cnt += 1
        img_name = str(cnt) + '_' + str(labels.item()) + '.png'
        # utils.save_tensor_images(imgs, os.path.join(mnist_img_path, img_name))
    print("number of train files:", cnt)

    for imgs, labels in test_loader:
        cnt += 1
        img_name = str(cnt) + '_' + str(labels.item()) + '.png'
        # utils.save_tensor_images(imgs, os.path.join(mnist_img_path, img_name))

def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(cifar_path, train=True, transform=transform, download=True)
    testset = torchvision.datasets.CIFAR10(cifar_path, train=False, transform=transform, download=True)

    train_loader = DataLoader(trainset, batch_size=1)
    test_loader = DataLoader(testset, batch_size=1)
    cnt = 0

    for imgs, labels in train_loader:
        cnt += 1
        img_name = str(cnt) + '_' + str(labels.item()) + '.png'
        utils.save_tensor_images(imgs, os.path.join(cifar_img_path, img_name))
    cnt_train = cnt
    print("number of train files:", cnt_train)

    for imgs, labels in test_loader:
        cnt += 1
        img_name = str(cnt) + '_' + str(labels.item()) + '.png'
        utils.save_tensor_images(imgs, os.path.join(cifar_img_path, img_name))

    print("number of test files:", cnt-cnt_train)


if __name__ == "__main__":
    # load_cifar10()
    print("ok")