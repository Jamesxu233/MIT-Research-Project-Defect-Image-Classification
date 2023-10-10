import glob
import numpy as np
import torch
import os
import cv2
from unet_model import UNet
from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=3, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('C:/Users/xukeh/Desktop/学习/MIT科研/课题四/K折交叉验证训练代码/results/K2_model_best.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    img_name_list = glob.glob('E:/data/new_dataset/test/*.png')
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list, lbl_name_list = [],transform=transforms.Compose([RescaleT(256),ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1,shuffle=False,num_workers=1)
    # 遍历所素有图片
    for i_test, data_test in enumerate(test_salobj_dataloader):
        # 保存结果地址
        save_res_path = img_name_list[i_test].split('.')[0] + '_res.png'
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        inputs_test = inputs_test.to(device=device, dtype=torch.float32)
        # 预测
        with torch.no_grad():
            pred = net(inputs_test)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        cv2.imwrite(save_res_path, pred)