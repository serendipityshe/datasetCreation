import os 
from tqdm import tqdm
import random
import cv2
import numpy as np
import pyhocon




class aug:
    # 图像增强
    def __init__(self, ImP: str, LaP: str, ExImP: str, ExLaP: str, file_format: str, flag = 'yolo_txt') -> None:
        self.ImagePath = ImP
        self.LabelPath = LaP
        self.ExportImagePath = ExImP
        self.ExportLabelPath = ExLaP
        self.FileFormat = file_format
        self.flag = flag


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
    
#-----------------------------------------------Label Conversion-------------------------------------------------------------

    def __rotate_and_scale_point(self, x, y, cx, cy, cos, sin, d):
        '''
        旋转图片对应的label变换
        Args:
            x: x坐标
            y: y坐标
            cos: 
            sin: 
            d: 
        '''
        if self.flag != 'yolo_txt':
            new_x = (x - cx) * cos - (y - cy) * sin + cx * d
            new_y = (y - cy) * cos + (x - cx) * sin + cy * d
        else:
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
            new_x1, new_y1 = self.__rotate_and_scale_point(x1, y1, cos, sin, d)
            new_x2, new_y2 = self.__rotate_and_scale_point(x2, y2, cos, sin, d)
            new_x3, new_y3 = self.__rotate_and_scale_point(x3, y3, cos, sin, d)
            new_x4, new_y4 = self.__rotate_and_scale_point(x4, y4, cos, sin, d)
            
            # 构建并添加变换后的边界框字符串到bbox列表
            space_separated_string = ' '.join(map(str, [class_id, new_x1, new_y1, new_x2, new_y2, new_x3, new_y3, new_x4, new_y4]))
            bbox.append(space_separated_string)
        
        return bbox
    
    def getBbox2MirrorHorizon(self, txt_path):
        bbox = []
        bboxes = self.get_bboxes(txt_path)
        for bbox_data in bboxes:
            class_id, x1, y1, x2, y2, x3, y3, x4, y4 = bbox_data[:]
            space_separated_string = ''.join(map(str, [class_id, 1 - x1, y1, 1 - x2, y2, 1 - x3, y3, 1 - x4, y4]))
            bbox.append(space_separated_string)
        return bbox
    



    def label2txt(self, labelInfo, txtPath):
        with open(txtPath, 'w') as f:
            f.writelines([line + os.linesep for line in labelInfo])


#--------------------------------------------Image Conversion------------------------------------------------------

    def Rotate(self, angle=45, ratio=1.0):
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
            labelInfor = self.getBbox2MirrorHorizon(self.LabelPath + '/' + name_only + '.txt')
            self.label2txt(labelInfor, self.ExportLabelPath + '/' + name_only + flag + '.txt')


            
            






if __name__ == "__main__":
    # aug = aug(r'D:\project\datasetCreation\data\daylightingBand\images\train',
    #           r'D:\project\datasetCreation\data\daylightingBand\labels\train',
    #           r'D:\project\datasetCreation\data\daylightingBand\images\train_ex',
    #           r'D:\project\datasetCreation\data\daylightingBand\labels\train_ex')
    
    # aug.Rotate()

    conf = pyhocon.ConfigFactory.parse_file('utils/dataEnhancement/conf/format.conf')
    w = conf.get('format_file')
    print(w)