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
    def __init__(self, ImP: str, ExImP: str, labelImage: str, EXLImP: str, img2: str, out2: str) -> None:
        self.ImagePath = ImP
        self.ExportImagePath = ExImP
        self.LabelImagePath = labelImage
        self.ExportLABELImgPath = EXLImP
        self.img2 = img2
        self.out2 = out2

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

    def MirrorHorizon(self, ratio=1.0):
        '''水平镜像'''
        flag = '001'
        image_files = self.get_file_name(self.ImagePath)
        image2_files = self.get_file_name(self.img2)
        label_files = self.get_file_name(self.LabelImagePath)

        def process_images(files, input_path, output_path):
            for filename in tqdm(files):
                name_only = os.path.splitext(os.path.basename(filename))[0]
                input_file_path = os.path.join(input_path, filename)
                output_file_path = os.path.join(output_path, name_only + flag + '.png')
                try:
                    image = cv2.imread(input_file_path)
                    if image is None:
                        raise FileNotFoundError(f"Image not found: {input_file_path}")
                    height, width, _ = image.shape
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    flipped = cv2.flip(image, 1)
                    flipped_bgr = cv2.cvtColor(flipped, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_file_path, flipped_bgr)
                except Exception as e:
                    print(f"Error processing file {input_file_path}: {e}")

        if ratio >= 1.0 or random.random() < ratio:
            process_images(label_files, self.LabelImagePath, self.ExportLABELImgPath)
            process_images(image_files, self.ImagePath, self.ExportImagePath)
            process_images(image2_files, self.img2, self.out2)

    def MirrorVertical(self, ratio = 1.0):
        '''垂直镜像
        Args: 
            ratio: 数据增强的概率
        Returns:
            None
        '''
        flag = '010'
        image_files = self.get_file_name(self.ImagePath)
        image2_files = self.get_file_name(self.img2)
        label_files = self.get_file_name(self.LabelImagePath)

        def process_images(files, input_path, output_path):
            for filename in tqdm(files):
                random_foloat = random.uniform(0, 1)
                if ratio < random_foloat:
                    continue
                name_only = os.path.splitext(os.path.basename(filename))[0]
                image = cv2.imread(input_path + '/' + filename)
                height, width, _ = image.shape
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                filpped = cv2.flip(image, 0)
                filpped_bgr = cv2.cvtColor(filpped, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path + '/' + name_only + flag + '.png', filpped_bgr)
        
        if ratio >= 1.0 or random.random() < ratio:
            process_images(image_files, self.ImagePath, self.ExportImagePath)
            process_images(label_files, self.LabelImagePath, self.ExportLABELImgPath)
            process_images(image2_files, self.img2, self.out2)


    def MirrorHV(self, ratio = 1.0):
        '''水平垂直镜像
        Args:
            ratio: 数据增强的概率
        Returns:
            None
        '''
        flag = '011'
        image_files = self.get_file_name(self.ImagePath)
        image2_files = self.get_file_name(self.img2)
        label_files = self.get_file_name(self.LabelImagePath)

        def process_images(files, input_path, output_path):
            for filename in tqdm(files):
                random_float = random.uniform(0, 1)
                if ratio < random_float:
                    continue
                name_only = os.path.splitext(os.path.basename(filename))[0]
                image = cv2.imread(input_path + '/' + filename)
                height, width, _ = image.shape
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                filpped = cv2.flip(image, -1)
                filpped_bgr = cv2.cvtColor(filpped, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path + '/' + name_only + flag + '.png', filpped_bgr)

        if ratio >= 1.0 or random.random() < ratio:
            process_images(image_files, self.ImagePath, self.ExportImagePath)
            process_images(label_files, self.LabelImagePath, self.ExportLABELImgPath)
            process_images(image2_files, self.img2, self.out2)



    def AddWeather(self, ratio = 1):
        '''添加噪声(AddWeather:对文件夹中的图片进行天气增强 1:1:1:1=雨天:雪天:日光:阴影)
        Args: 
            ratio: 数据增强的概率
        '''
        flag = '000'
        image_files = self.get_file_name(self.ImagePath)
        image2_files = self.get_file_name(self.img2)
        label_files = self.get_file_name(self.LabelImagePath)
        
        def process_images(files, input_path, output_path):
            for filename in tqdm(files):
                random_float = random.uniform(0, 1)
                if ratio < random_float:
                    continue
                name_only = os.path.splitext(os.path.basename(filename))[0]
                image = cv2.imread(input_path + '/' + filename)
                cv2.imwrite(output_path + '/' + name_only + flag + '.png', image)

        for filename in tqdm(image_files):
            random_float = random.uniform(0, 1)
            if ratio < random_float:
                continue
            name_only = os.path.splitext(os.path.basename(filename))[0]
            image = cv2.imread(self.ImagePath + '/' + filename)
            height, width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # random_number = random.randint(0, 3)
            random_number = 1
            if random_number == 0:
                transform = A.Compose(
                    [A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1)],
                )
            elif random_number == 1:
                transform = A.Compose(
                    [A.RandomSnow(brightness_coeff=2, snow_point_lower=0.3, snow_point_upper=0.5, p=1)],
                )
            elif random_number == 2:
                brightness_limit = random.uniform(0.1, 0.2)  # 亮度调整范围，可以根据需要调整
                contrast_limit = random.uniform(0.1, 0.52)  # 对比度调整范围，可以根据需要调整
                transform = A.Compose(
                    [A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=1)],
                )
            elif random_number == 3:
                var_limit = (200, 255)  # 高斯噪声的方差范围，可以根据需要调整
                transform = A.Compose(
                    [A.GaussNoise(var_limit=var_limit, p=1)],
                )
            transformed = transform(image=image)
            # Convert the transformed image back to a PIL image
            transformed_pil = Image.fromarray(transformed['image'])

            # Save the transformed image as a TIF image
            transformed_pil.save(self.ExportImagePath + '/' + name_only + flag + '.png')
        process_images(label_files, self.LabelImagePath, self.ExportLABELImgPath)
        process_images(image2_files, self.img2, self.out2)



if __name__ == "__main__":
    aug = aug(ImP=r'E:\xunlei\EIP-CD512\B', ExImP=r'E:\xunlei\EIP-CD512\snowB\B',
              img2=r'E:\xunlei\EIP-CD512\A', out2=r'E:\xunlei\EIP-CD512\snowB\A', 
              labelImage=r'E:\xunlei\EIP-CD512\label', EXLImP=r'E:\xunlei\EIP-CD512\snowB\label')

    aug.AddWeather()


