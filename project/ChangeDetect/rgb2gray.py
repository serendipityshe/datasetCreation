import numpy as np
import os
import cv2
import time
 
 
def color2gray(img_path, color_map):
    # 读取图片
    color_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # 计算时间
    t0 = time.time()
    gray_img = np.zeros(shape=(color_img.shape[0], color_img.shape[1]), dtype=np.uint8)
    for i in range(color_map.shape[0]):
        index = np.where(np.all(color_img == color_map[i], axis=-1))  # np.all true false
        gray_img[index] = i
    t1 = time.time()
    time_cost = round(t1 - t0, 3)
    print(f"color2label  cost time {time_cost}")
    # 保存图片
    dir, name = os.path.split(img_path)
    save_dir=r'E:\xunlei\EIP-SCD512_EN\label1_1'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, name)
    cv2.imwrite(save_path, gray_img)
 
if __name__ == '__main__':
    # 你的colormap  注意：这个是BGR的！！！！！！很重要
    cmap = np.array(
        [
            (0, 0, 0),
            (0,128,0), 
            (128,128,128), 
            (0,255,0), 
            (255,0,0), 
            (0,0,128), 
            (0,0,255)
        ]
    )
    # 文件路径
    img_dir = r'E:\xunlei\EIP-SCD512_EN\label1_rgb'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for img in os.listdir(img_dir):
        if not img.endswith((".png", ".jpg")):
            continue
        img_path = os.path.join(img_dir, img)
        color2gray(img_path, color_map=cmap)