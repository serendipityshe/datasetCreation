import os
from PIL import Image

def process_image(input_path, output_path):
    """
    处理单张图片，将所有不为0的像素转为255，并保存到指定路径。
    """
    with Image.open(input_path) as img:
        # 将图片转换为灰度图像（如果原图不是灰度图像）
        if img.mode != 'L':
            img = img.convert('L')
        
        # 获取图片的像素数据
        pixels = img.load()
        
        # 遍历每个像素并处理
        for x in range(img.width):
            for y in range(img.height):
                if pixels[x, y] != 0:
                    pixels[x, y] = 255
        
        # 保存处理后的图片
        img.save(output_path)

def process_all_images_in_folder(input_folder, output_folder):
    """
    处理指定文件夹下的所有图片，并将处理后的图片保存到另一个文件夹。
    """
    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 构建输入和输出文件的完整路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # 检查文件是否为图片（通过扩展名简单判断）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            # 处理图片
            process_image(input_path, output_path)
            print(f"Processed {input_path} and saved to {output_path}")

# 使用示例
input_folder = r'E:\xunlei\EIP-SCD512_EN\label1_rgb'
output_folder = r'E:\xunlei\EIP-SCD512_EN\label_CD'
process_all_images_in_folder(input_folder, output_folder)