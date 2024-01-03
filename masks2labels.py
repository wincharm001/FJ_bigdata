import os
import shutil
import cv2

masks_dir = './temp/'
labels_dir = './labels'
def masks2labels(masks_dir, labels_dir):
    for j in os.listdir(masks_dir):
        image_path = os.path.join(masks_dir, j)
        # load the binary mask and get its contours
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)  # 将灰度图像进行二值化

        H, W = mask.shape
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # convert the contours to polygons
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:  # 过滤掉面积小于200的轮廓
                polygon = []
                for point in cnt:
                    # 遍历轮廓的每个点坐标
                    x, y = point[0]
                    # 归一化坐标
                    polygon.append(x / W)
                    polygon.append(y / H)
                polygons.append(polygon)

        # print the polygons
        with open('{}.txt'.format(os.path.join(labels_dir, j)[:-4]), 'w') as f:
            for polygon in polygons:
                for p_, p in enumerate(polygon):
                    if p_ == len(polygon) - 1:
                        f.write('{}\n'.format(p))
                    elif p_ == 0:
                        f.write('0 {} '.format(p))
                    else:
                        f.write('{} '.format(p))

            f.close()


def split():
    with open("train.txt", 'r') as f:
        train_index = f.readlines()
        f.close()
    with open("val.txt", 'r') as f:
        val_index = f.readlines()
        f.close()
    
    train_index = [int(i[:-1]) for i in train_index]
    val_index = [int(i[:-1]) for i in val_index]
    
    labels_names = os.listdir(labels_dir)

    for i in range(len(labels_names)):
        index = int(labels_names[i].split('_')[0])
        if index in train_index:
            shutil.move(f'labels/{index}_mask.txt', f'datasets/labels/train/{index}.txt')
        elif index in val_index:
            shutil.move(f'labels/{index}_mask.txt', f'datasets/labels/val/{index}.txt')
        else:
            print('error!')
            exit(0)


masks2labels(masks_dir, labels_dir)
split()