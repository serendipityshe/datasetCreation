from PIL import Image
import os

def convert_image_pixels(image_path, output_dir):
    # 打开图片
    image = Image.open(image_path)
    
    # 转换图片模式为RGB（如果需要）
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 获取图片的像素数据
    pixels = image.load()
    
    # 遍历图片的每个像素，将像素值为255的转换为1
    width, height = image.size
    for x in range(width):
        for y in range(height):
            if pixels[x, y] == (255, 255, 255):  # 检查是否为白色
                pixels[x, y] = (1, 1, 1)  # 转换为像素值1
    
    # 保存修改后的图片到输出目录
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, base_name)
    image.save(output_path)

def process_images_in_directory(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历输入目录下的所有文件
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        
        # 检查文件是否是一个图片文件（根据扩展名）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            convert_image_pixels(file_path, output_dir)

# 指定输入和输出目录
input_directory = r'D:\PROJECT\AI-Project\ChangeDetect\SCanNet\datasets\SECOND\label_CD'  # 替换为您的输入目录路径
output_directory = r'D:\PROJECT\AI-Project\ChangeDetect\SCanNet\datasets\SECOND\label_CDBI'  # 替换为您的输出目录路径

# 处理输入目录下的所有图片
process_images_in_directory(input_directory, output_directory)
print("图片处理完成。")