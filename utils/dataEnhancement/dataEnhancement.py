import os 
from tqdm import tqdm
import random
import cv2
import numpy as np
import pyhocon
import albumentations as A
import shutil
import time
from PIL import Image
import skimage


class aug:
    # 图像增强
    def __init__(self, ImP: str, LaP: str, ExImP: str, ExLaP: str, file_format: str = 'yolo_txt') -> None:
        self.ImagePath = ImP
        self.LabelPath = LaP
        self.ExportImagePath = ExImP
        self.ExportLabelPath = ExLaP
        self.FileFormat = file_format

#--------------------------------------------------Foundation Work--------------------------------------------------------------

    def get_file_name(self, path):
        '''
        获取目录下的文件名字(带后缀)
        Args:
            path: 文件路径
        '''
        file_names = []
        for file_entry in os.scandir(path):
            if file_entry.is_file():
                file_names.append(file_entry.name)
        return file_names

    def get_bboxes(self, txt_file_path):
        '''
        获取yolo_txt文件中的矩形框坐标
        Args:
            txt_file_path: txt文件路径
        '''
        with open(txt_file_path, 'r') as f:
            lines = f.readlines()
        bboxes = []
        for line in lines:
            class_id, x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line.split())
            bboxes.append([class_id, x1, y1, x2, y2, x3, y3, x4, y4])
        return bboxes
    
    
    def label2txt(self, labelInfo, txtPath):
        with open(txtPath, 'w') as f:
            f.writelines([line + os.linesep for line in labelInfo])

#-----------------------------------------------Label Conversion-------------------------------------------------------------
# 1. 旋转
# 2. 水平镜像
# 3. 垂直镜像
# 4. 水平垂直镜像
# 5. 随机裁剪
# 6. 添加噪声（添加天气情况）

    def __rotate_and_scale_point(self, x, y, cx, cy, cos, sin, d):
        '''辅助函数： 旋转图片对应的label变换
        Args:
            x: x坐标
            y: y坐标
            cos: 
            sin: 
            d: 
        '''
        new_x = (x - 0.5) * cos - (y - 0.5) * sin + 0.5 * d
        new_y = (y - 0.5) * cos + (x - 0.5) * sin + 0.5 * d
        return new_x, new_y


    def getBboxRotate(self, txt_file_path, angle, w, h, d):
        '''旋转图像和坐标
        '''
        bbox = []
        bboxes = self.get_bboxes(txt_file_path)
        cx = w / 2
        cy = h / 2
        cos = np.cos(np.deg2rad(-angle))  # 修正了角度，使其符合常规的顺时针旋转
        sin = np.sin(np.deg2rad(-angle))  # 同上
        for bbox_data in bboxes:
            class_id, x1, y1, x2, y2, x3, y3, x4, y4 = bbox_data[:]
            
            # 对每个点进行旋转和缩放变换
            new_x1, new_y1 = self.__rotate_and_scale_point(x1, y1, cx, cy, cos, sin, d)
            new_x2, new_y2 = self.__rotate_and_scale_point(x2, y2, cx, cy, cos, sin, d)
            new_x3, new_y3 = self.__rotate_and_scale_point(x3, y3, cx, cy, cos, sin, d)
            new_x4, new_y4 = self.__rotate_and_scale_point(x4, y4, cx, cy, cos, sin, d)
            
            # 构建并添加变换后的边界框字符串到bbox列表
            space_separated_string = ' '.join(map(str, [class_id, new_x1, new_y1, new_x2, new_y2, new_x3, new_y3, new_x4, new_y4]))
            bbox.append(space_separated_string)
        
        return bbox
    
    def getBbox2MirrorHorizon(self, txt_path):
        '''水平镜像变换
        Args:
            txt_path: txt文件路径
        Returns:
            bbox: 变换后的边界框列表
        '''
        bbox = []
        bboxes = self.get_bboxes(txt_path)
        for bbox_data in bboxes:
            class_id, x1, y1, x2, y2, x3, y3, x4, y4 = bbox_data[:]
            space_separated_string = ' '.join(map(str, [class_id, 1 - x1, y1, 1 - x2, y2, 1 - x3, y3, 1 - x4, y4]))
            bbox.append(space_separated_string)
        return bbox
    
    def getBbox2Vertical(self, txt_path):
        '''垂直镜像变换
        Args:
            txt_path: txt文件路径
        Returns:
            bbox: 变换后的边界框列表
        '''
        bbox = []
        bboxes = self.get_bboxes(txt_path)
        for bbox_data in bboxes:
            class_id, x1, y1, x2, y2, x3, y3, x4, y4 = bbox_data[:]
            space_separated_string = ' '.join(map(str, [class_id, x1, 1 - y1, x2, 1 - y2, x3, 1 - y3, x4, 1 - y4]))
            bbox.append(space_separated_string)
        return bbox
    
    def getBbox2HV(self, txt_path):
        '''水平垂直镜像变换
        Args:
            txt_path: txt文件路径
        Returns:
            bbox: 变换后的边框列表
        ''' 
        bbox = []
        bboxes = self.get_bboxes(txt_path)
        for bbox_data in bboxes:
            class_id, x1, y1, x2, y2, x3, y3, x4, y4 = bbox_data[:]
            space_separated_string = ' '.join(map(str, [class_id, 1 - x1, 1 - y1, 1 - x2, 1 - y2, 1 - x3, 1 - y3, 1 - x4, 1 - y4]))
            bbox.append(space_separated_string)
        return bbox

    def getBbox2Resize(self, txt_path, scale, left, top):
        '''随机裁剪
        Args:
            txt_path: txt文件路径
            scale: 缩放比例
            left: 左边距
            top: 顶部边距
        Returns:
            bbox: 变换后的边框列表
        '''
        bbox = []
        bboxes = self.get_bboxes(txt_path)
        for bbox_data in bboxes:
            class_id, x1, y1, x2, y2, x3, y3, x4, y4 = bbox_data[:]
            space_separated_string = ' '.join(map(str, [class_id, x1*scale+left, y1*scale+top, x2*scale+left, y2*scale+top, 
                                                       x3*scale+left, y3*scale+top, x4*scale+left, y4*scale+top]))
            bbox.append(space_separated_string)
        return bbox

 

#--------------------------------------------Image Conversion------------------------------------------------------
# 1. 旋转
# 2. 水平镜像
# 3. 垂直镜像
# 4. 水平垂直镜像
# 5. 随机裁剪
# 6. 添加噪声（添加天气情况）


    def Rotate(self, angle=45, ratio=1.0):
        '''旋转图像和坐标
        Args:
            angle: 旋转角度
            ratio: 数据增强的概率
        Returns:
            None
        '''
        flag = '100'
        Filelist = self.get_file_name(self.ImagePath)
        for filename in tqdm(Filelist):
            random_float = random.uniform(0, 1)
            if ratio < random_float:
                continue
            name_only = os.path.splitext(os.path.basename(filename))[0]
            image = cv2.imread(self.ImagePath + '/' + filename)
            height, width, _ = image.shape
            center = (width / 2, height / 2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            matrix = cv2.getRotationMatrix2D(center, angle, 1)

            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))

            matrix[0, 2] += (new_width - width) / 2
            matrix[1, 2] += (new_height - height) / 2

            rotated = cv2.warpAffine(image, matrix, (new_width, new_height), borderValue=(0, 0, 0))
            
            rotated_bgr = cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)

            cv2.imwrite(self.ExportImagePath + '/' + name_only + flag + '.tif', rotated_bgr)
            labelInfo = self.getBboxRotate(self.LabelPath + '/' + name_only + '.txt', angle, width, height, new_height / height)
            self.label2txt(labelInfo, self.ExportLabelPath + '/' + name_only + flag + '.txt')


    def MirrorHorizon(self, ratio = 1.0):
        '''水平镜像
        '''
        flag = '001'
        FileList = self.get_file_name(self.ImagePath)
        for filename in tqdm(FileList):
            random_float = random.uniform(0, 1)
            if ratio < random_float:
                continue
            name_only = os.path.splitext(os.path.basename(filename))[0]
            image = cv2.imread(self.ImagePath + '/' + filename)
            height, width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            flipped = cv2.flip(image, 1)

            flipped_bgr = cv2.cvtColor(flipped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.ExportImagePath + '/' + name_only + flag + '.tif', flipped_bgr)
            labelInfo = self.getBbox2MirrorHorizon(self.LabelPath + '/' + name_only + '.txt')
            self.label2txt(labelInfo, self.ExportLabelPath + '/' + name_only + flag + '.txt')


    def MirrorVertical(self, ratio = 1.0):
        '''垂直镜像
        Args: 
            ratio: 数据增强的概率
        Returns:
            None
        '''
        flag = '010'
        FileList = self.get_file_name(self.ImagePath)
        for filename in tqdm(FileList):
            random_foloat = random.uniform(0, 1)
            if ratio < random_foloat:
                continue
            name_only = os.path.splitext(os.path.basename(filename))[0]
            image = cv2.imread(self.ImagePath + '/' + filename)
            height, width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            filpped = cv2.flip(image, 0)
            filpped_bgr = cv2.cvtColor(filpped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.ExportImagePath + '/' + name_only + flag + '.tif', filpped_bgr)
            labelInfo = self.getBbox2Vertical(self.LabelPath + '/' + name_only + '.txt')
            self.label2txt(labelInfo, self.ExportLabelPath + '/' + name_only + flag + '.txt')


    def MirrorHV(self, ratio = 1.0):
        '''水平垂直镜像
        Args:
            ratio: 数据增强的概率
        Returns:
            None
        '''
        flag = '011'
        FileList = self.get_file_name(self.ImagePath)
        for filename in tqdm(FileList):
            random_float = random.uniform(0, 1)
            if ratio < random_float:
                continue
            name_only = os.path.splitext(os.path.basename(filename))[0]
            image = cv2.imread(self.ImagePath + '/' + filename)
            height, width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            filpped = cv2.flip(image, -1)
            filpped_bgr = cv2.cvtColor(filpped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.ExportImagePath + '/' + name_only + flag + '.tif', filpped_bgr)
            labelInfo = self.getBbox2HV(self.LabelPath + '/' + name_only + '.txt')
            self.label2txt(labelInfo, self.ExportLabelPath + '/' + name_only + flag + '.txt')


    def RandomCrop(self, min_scale = 0.3, max_scale = 0.7, ratio = 1.0):
        '''随机裁剪

        '''
        flag = '200'
        FileList = self.get_file_name(self.ImagePath)
        for filename in tqdm(FileList):
            random_foloat = random.uniform(0, 1)
            if ratio < random_foloat:
                continue
            name_only = os.path.splitext(os.path.basename(filename))[0]
            image = cv2.imread(self.ImagePath + '/' + filename)
            height, width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            random_scale = random.uniform(min_scale, max_scale)

    
    def AddWeather(self, ratio = 1):
        '''添加噪声(AddWeather:对文件夹中的图片进行天气增强 1:1:1:1=雨天:雪天:日光:阴影)
        Args: 
            ratio: 数据增强的概率
        '''
        flag = '000'
        FileList = self.get_file_name(self.ImagePath)
        for filename in tqdm(FileList):
            random_float = random.uniform(0, 1)
            if ratio < random_float:
                continue
            name_only = os.path.splitext(os.path.basename(filename))[0]
            image = cv2.imread(self.ImagePath + '/' + filename)
            height, width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # random_number = random.randint(0, 3)
            random_number = 3
            if random_number == 0:
                transform = A.Compose(
                    [A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1)],
                )
            elif random_number == 1:
                transform = A.Compose(
                    [A.RandomSnow(brightness_coeff=1.2, snow_point_lower=0.3, snow_point_upper=0.5, p=1)],
                )
            elif random_number == 2:
                brightness_limit = random.uniform(-0.2, 0.2)  # 亮度调整范围，可以根据需要调整
                contrast_limit = random.uniform(0.9, 1.1)  # 对比度调整范围，可以根据需要调整
                transform = A.Compose(
                    [A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=1)],
                )
            elif random_number == 3:
                var_limit = (10, 255)  # 高斯噪声的方差范围，可以根据需要调整
                transform = A.Compose(
                    [A.GaussNoise(var_limit=var_limit, p=1)],
                )
            transformed = transform(image=image)
            # Convert the transformed image back to a PIL image
            transformed_pil = Image.fromarray(transformed['image'])

            # Save the transformed image as a TIF image
            transformed_pil.save(self.ExportImagePath + '/' + name_only + flag + '.tif')

            shutil.copy(self.LabelPath + '/' + name_only + '.txt', self.ExportLabelPath + '/' + name_only + flag + '.txt')


if __name__ == "__main__":
    aug = aug(r'D:\DAS_DATASET\data\test\datasets\train\images',
              r'D:\DAS_DATASET\data\test\datasets\train\labels',
              r'D:\DAS_DATASET\data\test\datasets\train\images\images_ex',
              r'D:\DAS_DATASET\data\test\datasets\train\labels\labels_ex')
    
    # aug.Rotate()
    # aug.MirrorHV()
    aug.AddWeather()
    # aug.RandomCrop()
    # aug.MirrorVertical()
    # aug.MirrorHorizon()
