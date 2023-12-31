import torch
import torchvision
from PIL import Image
import data_loader
import csv

class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, is_train,root):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(root, 'r',newline='')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        fh_reader = csv.reader(fh)
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh_reader:  # 按行循环txt文本中的内容
            #print(line)
            imgs.append((line[0], line[1]))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
        self.imgs = imgs
        self.is_train = is_train
        if self.is_train:
            self.train_tsf = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.ToTensor()
            ])
        else:
            self.test_tsf = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.ToTensor()])

    def __getitem__(self,idx):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        feature, label = self.imgs[idx] # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        feature = Image.open(feature).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        label= Image.open(label)
        if self.is_train:
            feature = self.train_tsf(feature)
            label = self.train_tsf(label)
        else:
            feature = self.test_tsf(feature)
            label = self.test_tsf(label)
        return feature, label

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
