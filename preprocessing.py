from PIL import Image
from PIL import ImageFile
import random
from tqdm import tqdm


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


img1 = train_imgs[0]
img1_mask = train_masks[0]
# 旋转img2，裁掉黑边
img2 = train_imgs[1].rotate(angle=13)
img2_mask = train_masks[1].rotate(angle=13)


(left, upper, right, lower) = (2992.0, 1500.0, 15932.0, 22680.0)
img2 = img2.crop((left, upper, right, lower))
img2_mask = img2_mask.crop((left, upper, right, lower))
# img2.save(train_path + "img2/img2_rotated_cropped.tif")
# img2_mask.save(train_path + "img2/img2_mask_rotated_cropped.tif")

# img1: (22019, 11369), img2: (12940, 21180) -> (640, 640)
crop_size = (640, 640)

def crop_img(img: Image, width_overlap=160):
    # 将两张图片裁剪成相同形状的一组图像
    # 重叠 160 的宽度（根据可能的最小渔排的大小和图像裁剪大小来确定）----> TODO: 80 and only train
    (width, height) = img.size  # (img.width, img.height)
    img_list = []

    num_cols, remainder_x = width // crop_size[0], width % crop_size[0]
    num_rows, remainder_y = height // crop_size[1], height % crop_size[1]
    remainder = 80  # 根据可能的最小渔排的大小来确定
    if remainder_x >= remainder and remainder_y >= remainder:
        img = img.resize(((num_cols + 1) * crop_size[0], (num_rows + 1) * crop_size[1]))

    elif remainder_x >= remainder and remainder_x < remainder:
        img = img.resize(((num_cols + 1) * crop_size[0], num_rows * crop_size[1]))

    elif remainder_x < remainder and remainder_x >= remainder:
        img = img.resize((num_cols * crop_size[0], (num_rows + 1) * crop_size[1]))

    else:
        img = img.resize((num_cols * crop_size[0], num_rows * crop_size[1]))

    (new_w, new_h) = img.size
    new_img = Image.new('RGB', size=(new_w + width_overlap * 2, new_h + width_overlap * 2))
    new_nc, new_nr = new_w // (crop_size[0]), new_w // (crop_size[1])

    new_img.paste(img, box=(width_overlap, width_overlap))
    # new_img.show()
    top = 0
    for i in tqdm(range(new_nr)):
        left = 0
        bottom = top + crop_size[0]
        for j in range(new_nc):
            right = left + crop_size[1]
            cropped_img = new_img.crop((left, top, right, bottom))
            img_list.append(cropped_img)
            left = right - width_overlap  # 更新下一小块图像的左边界

        top = bottom - width_overlap


    return img_list


img1_list = crop_img(img1)
img1_mask_list = crop_img(img1_mask)
img2_list = crop_img(img2)
img2_mask_list = crop_img(img2_mask)

img_list = img1_list + img2_list
img_mask_list = img1_mask_list + img2_mask_list


def train_val_split(X, test_size=0.3):
    '''生成从 0 到 len(X) 的序列，打乱序列，将序列最后的那部分作为
       验证集，返回打乱后的训练集和验证集的序号'''
    random.seed(2024)
    nums = len(X)
    train_index = [i for i in range(nums)]
    test_num = round(nums * test_size)  # 验证集的数据个数
    random.shuffle(train_index)
    val_index = train_index[(nums - test_num): nums]
    train_index = train_index[0: (nums - test_num)]
    return train_index, val_index

# train_index, val_index = train_val_split(img_list, test_size=0.3)

# # 存储训练集与验证集的样本序号
# with open("train2.txt", 'w') as f:
#     for i in train_index:
#         f.write(str(i) + '\n')

# with open("val2.txt", 'w') as f:
#     for i in val_index:
#         f.write(str(i) + '\n')


# for i in tqdm(train_index):
#     img_list[i].save(f"datasets/images/train2/{i}.png")
#     img_mask_list[i].save(f"datasets/masks/train2/{i}_mask.png")

# for i in tqdm(val_index):
#     img_list[i].save(f"datasets/images/val2/{i}.png")
#     img_mask_list[i].save(f"datasets/masks/val2/{i}_mask.png")


for i in range(len(img_list)):
    img_list[i].save(f"datasets/images/train3/{i}.png")
    img_mask_list[i].save(f"datasets/masks/train3/{i}_mask.png")