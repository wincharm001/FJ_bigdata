from PIL import Image
from PIL import ImageFile
import numpy as np
import torch
import torchvision


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


train_path = "../toUser/train/"
train_imgs = []
train_masks = []

train_img = Image.open(train_path + "img1/img1.tif")
train_mask = Image.open(train_path + "img1/img1_mask.tif")  # img1_mask.tif 文件的mask前面多了一个空格
train_imgs.append(train_img)
train_masks.append(train_mask)


# 旋转img2，裁掉黑边
# img2 = train_imgs[1].rotate(angle=13)
# img2_mask = train_masks[1].rotate(angle=13)
# img2.save("../toUser/train/img2/img2_rotated.tif")
# img2_mask.save("../toUser/train/img2/img2_mask_rotated.tif")


# (left, upper, right, lower) = (2992.0, 1500.0, 15932.0, 22680.0)
# img2 = Image.open(train_path + "img2/img2_rotated.tif")
# img2_mask = Image.open(train_path + "img2/img2_mask_rotated.tif")
# img2 = img2.crop((left, upper, right, lower))
# img2_mask = img2_mask.crop((left, upper, right, lower))
# img2.save(train_path + "img2/img2_rotated_cropped.tif")
# img2_mask.save(train_path + "img2/img2_mask_rotated_cropped.tif")


img2 = Image.open(train_path + "img2/img2_rotated_cropped.tif")
img2_mask = Image.open(train_path + "img2/img2_mask_rotated_cropped.tif")
train_imgs.append(img2)
train_masks.append(img2_mask)


# for i in range(0, 2):
#     print(train_imgs[i].size)
#     print(train_masks[i].size)
# img1: (22019, 11369), img2: (12940, 21180)