from multiprocessing import reduction
from cv2 import mean
from unet_model import UNet
from dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import glob
from torch.autograd import Variable

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

import Dice


 
def train_net(net, device, tra_img_name_list, tra_lbl_name_list, epochs=40, batch_size=1, lr=0.00001):
    # 加载训练集
    salobj_dataset=SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(256),
            RandomCrop(224),
            ToTensorLab(flag=0)]))
    salobj_dataloader=DataLoader(salobj_dataset,batch_size=batch_size,shuffle=True,num_workers=1)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # best_loss统计，初始化为正无穷
    # 定义Loss算法
    criterion = Dice.SoftDiceLoss()
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for i, data in enumerate(salobj_dataloader):
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image,label = data['image'], data['label']
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道3，分类为2。
    net = UNet(n_channels=3, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    tra_img_name_list=glob.glob('E:/data/CRACK_dataset/train/*.jpg')
    tra_lbl_name_list=glob.glob('E:/data/CRACK_dataset/label/*.png')
    train_net(net, device, tra_img_name_list, tra_lbl_name_list)