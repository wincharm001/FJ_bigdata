from PIL import Image
from PIL import ImageFile
import numpy as np
import torch
import torchvision


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


traindata_path = "../toUser/train/"
train_imgs = []
train_masks = []
for i in range(1, 3):
    train_img = Image.open(traindata_path + f"img{i}/img{i}.tif")
    train_mask = Image.open(traindata_path + f"img{i}/img{i}_mask.tif")
    train_imgs.append(train_img)
    train_masks.append(train_mask)

# img1: [22019, 11369]  img2: [18164, 24004]

# TODO: 查看pillow打开的tif图像的通道顺序；将Image格式转为torch.Tensor格式；裁剪图像和mask，构建数据集
