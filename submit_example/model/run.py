import os, sys
import cv2
import random
import tifffile
import numpy as np
from PIL import Image, ImageFile
from ultralytics import YOLO

#! 突破大文件限制, 读取4GB以上tif文件
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None





def main(to_pred_dir, result_save_path):
    run_py_path = os.path.abspath(__file__)  #! run.py路径
    model_dir = os.path.dirname(run_py_path) #! model文件夹路径
    model = YOLO(model_dir + "model.pt")  # TODO

    pred_imgs_paths = os.listdir(to_pred_dir)
    pred_img_path = os.path.join(to_pred_dir, pred_imgs_paths[0]) #! 测试集只有一张图片
    to_pred_image = Image.open(pred_img_path) 
    print(to_pred_image.size) #! to_pred_image.size (w, h)
    image_array = np.array(to_pred_image)
    print(image_array.shape)  #! image_array.shape (h, w, c)
    #! pred = model(to_pred_image) 根据实际情况使用模型进行预测

    pred = model.predict(image_array)
    print(pred.shape) #! (h, w)

    #! cv保存
    cv2.imwrite(result_save_path, pred) #! 结果保存
    
    # #! PIL保存
    # pred = Image.fromarray(pred)
    # print(pred.size) #! (w, h)
    # pred.save(result_save_path) #! 结果保存

if __name__ == "__main__":
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径
    main(to_pred_dir, result_save_path)
