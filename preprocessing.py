from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


train_path = "../toUser/train/"
train_imgs = []
train_masks = []

for i in range(1, 3):
    train_img = Image.open(train_path + f"img{i}/img{i}.tif")
    train_mask = Image.open(train_path + f"img{i}/img{i}_mask.tif")  # img1_mask.tif 文件的mask前面多了一个空格
    train_imgs.append(train_img)
    train_masks.append(train_mask)


# 旋转img2，裁掉黑边
img2 = train_imgs[1].rotate(angle=13)
img2_mask = train_masks[1].rotate(angle=13)


(left, upper, right, lower) = (2992.0, 1500.0, 15932.0, 22680.0)
img2 = img2.crop((left, upper, right, lower))
img2_mask = img2_mask.crop((left, upper, right, lower))
img2.save(train_path + "img2/img2_rotated_cropped.tif")
img2_mask.save(train_path + "img2/img2_mask_rotated_cropped.tif")

# img1: (22019, 11369), img2: (12940, 21180)