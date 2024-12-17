import os
import random
import shutil

class ImageProcessor:
    def __init__(self, img_dir, root_dir=None):
        if root_dir is None:
            root_dir = os.path.dirname(os.path.abspath(__file__))
        self.img_dir = os.path.join(root_dir, img_dir)
        self.train_list_path = os.path.join(root_dir, 'train_list.txt')
        self.val_list_path = os.path.join(root_dir, 'val_list.txt')
        self.test_list_path = os.path.join(root_dir, 'test_list.txt')
        # 验证集和测试集的比例
        self.val_ratio = 1/5.0
        self.test_ratio = 1/5.0

    def process_images(self):
        img_list = os.listdir(self.img_dir)
        valid_list = [it for it in img_list if it.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
        num_pics = len(valid_list)
        num_val = int(num_pics * self.val_ratio)
        num_test = int(num_pics * self.test_ratio)
        num_train = num_pics - num_val - num_test

        print(f'num_pics: {num_pics}, num_train: {num_train}, num_val: {num_val}, num_test: {num_test}.')

        random.shuffle(valid_list)
        train_list, val_list, test_list = [], [], []
        for idx, it in enumerate(valid_list):
            if idx < num_train:
                train_list.append(it)
            elif idx < num_train + num_val:
                val_list.append(it)
            else:
                test_list.append(it)

        with open(self.train_list_path, 'w') as train_info:
            for it in train_list:
                train_info.write(it + '\n')
        with open(self.val_list_path, 'w') as val_info:
            for it in val_list:
                val_info.write(it + '\n')
        with open(self.test_list_path, 'w') as test_info:
            for it in test_list:
                test_info.write(it + '\n')

    def copy_images_to_dir(self, image_names_file, destination_dir):
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        with open(image_names_file, 'r', encoding='utf-8') as file:
            image_names = file.readlines()

        image_names = [name.strip() for name in image_names]

        for image_name in image_names:
            source_path = os.path.join(self.img_dir, image_name)
            destination_path = os.path.join(destination_dir, image_name)

            if os.path.isfile(source_path):
                shutil.copy2(source_path, destination_path)
                print(f"Copied {image_name} to {destination_path}")
            else:
                print(f"Source file {source_path} does not exist.")

# 使用示例
if __name__ == "__main__":
    processor = ImageProcessor('im1') # 图片源路径
    processor.process_images() # 生成训练、验证、测试集的列表文件
    processor.copy_images_to_dir(processor.train_list_path, r'D:\PROJECT\AI-Project\ChangeDetect\SCanNet\datasets\SECOND\ST\train\label1') # 将训练集、验证集、测试集的图片复制到对应的目录下
    processor.copy_images_to_dir(processor.val_list_path, r'D:\PROJECT\AI-Project\ChangeDetect\SCanNet\datasets\SECOND\ST\val\label1')
    processor.copy_images_to_dir(processor.test_list_path, r'D:\PROJECT\AI-Project\ChangeDetect\SCanNet\datasets\SECOND\ST\test\label1')