#本段代码为“k折交叉验证”提供了数据集中每张图片的的路径与标签信息
import glob
import os
import numpy as np
image_list=glob.glob('E:/data/new_dataset/train/image/*.png')
label_list=glob.glob('E:/data/new_dataset/train/label/*.png')
img_path=[]
sum=len(image_list)
#遍历上面的路径，依次把信息追加到img_path列表中
for i in range(sum):
    img_path.append((image_list[i],label_list[i]))
np.random.shuffle(img_path)
file=open("shuffle_data.txt","w",encoding="utf-8")
for img  in img_path:
    file.write(img[0]+','+img[1]+'\n')
file.close()

