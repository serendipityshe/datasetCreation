import os
import shutil

def copy_images(image_names_file, source_dir, destination_dir):
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # 读取txt文件中的图片名称
    with open(image_names_file, 'r', encoding='utf-8') as file:
        image_names = file.readlines()
    
    # 去除每行末尾的换行符
    image_names = [name.strip() for name in image_names]
    
    # 复制图片
    for image_name in image_names:
        source_path = os.path.join(source_dir, image_name)
        destination_path = os.path.join(destination_dir, image_name)
        
        # 检查源文件是否存在
        if os.path.isfile(source_path):
            shutil.copy2(source_path, destination_path)
            print(f"Copied {image_name} to {destination_path}")
        else:
            print(f"Source file {source_path} does not exist.")

# 示例用法
image_names_file = r'datasets\SECOND\test_list.txt'  # 图片名称的txt文件路径
source_dir = r'D:\PROJECT\AI-Project\ChangeDetect\SCanNet\datasets\SECOND\label2'          # 源图片目录
destination_dir = r'D:\PROJECT\AI-Project\ChangeDetect\open-cd\data\SECOND1\test\label2'  # 目标目录

copy_images(image_names_file, source_dir, destination_dir)