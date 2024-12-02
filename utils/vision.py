import cv2 
import os
import numpy as np
 
import cv2
import os
import numpy as np

thr = 0.95

def GetFileFromThisRootDir(dir, ext=None):
    allfiles = []
    for root, dirs, files in os.walk(dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            if ext is None or os.path.splitext(filepath)[1][1:] in ext:
                allfiles.append(filepath)
    return allfiles

def visualise_gt(label_path, pic_path, newpic_path):
    results = GetFileFromThisRootDir(label_path)
    for result in results:
        name = os.path.splitext(os.path.basename(result))[0]
        filepath = os.path.join(pic_path, f"{name}.jpg")
        im = cv2.imread(filepath)
        if im is None:
            print(f"图片文件 {filepath} 读取失败")
            continue
        im_h, im_w = im.shape[:2]

        with open(result, 'r') as f:
            lines = f.readlines()
        if not lines:
            print('文件为空', result)
            continue

        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 8:
                coordinates = [float(x) for x in parts[1:9]]
                for i in range(0, 8, 2):
                    coordinates[i] *= im_w  # 修复方式：将i的起始值改为0，确保索引不越界
                    coordinates[i+1] *= im_h
                boxes.append(coordinates)
        if not boxes:
            print('没有有效的框数据', result)
            continue

        boxes = np.array(boxes, dtype=np.float16)

        for box in boxes:
            box_coords = np.array([[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]], np.int32)
            box_coords = box_coords.reshape((-1, 1, 2))
            cv2.polylines(im, [box_coords], True, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(newpic_path, f"{name}.png"), im)

if __name__ == '__main__':
    pic_path = r"E:\test_data\test_data\images"
    label_path = r"E:\test_data\test_data\labels"
    newpic_path = 'vision/test2'
    if not os.path.isdir(newpic_path):
        os.makedirs(newpic_path)
    visualise_gt(label_path, pic_path, newpic_path)
 
# 
