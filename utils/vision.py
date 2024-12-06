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
        with open(result, 'r') as f:
            lines = f.readlines()
        if not lines:  # 如果为空
            print('文件为空', result)
            continue
        
        name = os.path.splitext(os.path.basename(result))[0]
        filepath = os.path.join(pic_path, f"{name}.jpg")
        im = cv2.imread(filepath)
        im_h = im.shape[0]
        im_w = im.shape[1]
        
        # 过滤掉空行和包含少于8个元素的行
        boxes = []
        for line in lines:
            parts = line.strip().split(' ')
            parts = list(map(lambda x: float(x)*512, filter(None, parts)))
            if len(parts) >= 8:
                boxes.append(np.array(parts[1:9], dtype=np.float64))
        if not boxes:
            print('没有有效的框数据', result)
            continue
        
        boxes = np.array(boxes)
        
 

        
        for box in boxes:
            box = np.array([[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]], np.int32)
            box = box.reshape((-1, 1, 2))
            cv2.polylines(im, [box], True, (0, 0, 255), 2)
        
        cv2.imwrite(os.path.join(newpic_path, f"{name}.png"), im)

if __name__ == '__main__':
    pic_path = r"E:\test_data\images"
    label_path = r"E:\test_data\labels"
    newpic_path = 'vision/test4'
    if not os.path.isdir(newpic_path):
        os.makedirs(newpic_path)
    visualise_gt(label_path, pic_path, newpic_path)
 
# 
