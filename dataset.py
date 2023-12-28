from PIL import Image
from PIL import ImageFile
import torch
import torchvision


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

train_path = "../toUser/train/"

img1 = Image.open(train_path + "img1/img1.tif")
img1_mask = Image.open(train_path + "img1/img1_mask.tif")
img2 = Image.open(train_path + "img2/img2.tif")
img2_mask = Image.open(train_path + "img2/img2_mask.tif")
# img1: (22019, 11369), img2: (12940, 21180) -> (800, 500)
SIZE = (800, 500)

def crop_tif(img: Image):
    img_list = []
    # 将两张图片裁剪成相同形状的一组图像
    (length, width) = img.size  # (img.width, img.height)
    for i in range(length // SIZE[0]):
        for j in range(width // SIZE[1]):
            img_cropped = img.crop((i * SIZE[0], j * SIZE[1], (i + 1) * SIZE[0], (j + 1) * SIZE[1]))
            img_list.append(img_cropped)

    return img_list


imgs = crop_tif(img1)
for i in range(10):
    print(imgs[i].size)

print(len(imgs))









# class DataSet(torch.utils.data.Dataset):
#     def __init__(self):


    
#     def __len__(self):
    

#     def __getitem__(self):