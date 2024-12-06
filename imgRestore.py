import os
from PIL import Image

def process_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', 'tif')):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    data = list(img.getdata())
                    
                    # 遍历像素数据，将像素值为1的改为255
                    new_data = [255 if pixel == 1 else pixel for pixel in data]
                    
                    img.putdata(new_data)
                    img.save(file_path)  # 保存修改后的图像，覆盖原文件
                    print(f"Processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# 使用示例
folder_path = r'E:\xunlei\TDOM\test'  # 替换为你的图片文件夹路径
process_images(folder_path)
