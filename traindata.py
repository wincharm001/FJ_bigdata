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
    train_mask = Image.open(traindata_path + f"img{i}/img{i}_mask.tif")  # img1_mask.tif 文件的mask前面多了一个空格
    train_imgs.append(train_img)
    train_masks.append(train_mask)

# img1: [22019, 11369]  img2: [18164, 24004]

# 旋转img2并裁剪
img2 = train_imgs[1].rotate(angle=13)
# 先在Image格式下裁剪，再转为tensor



