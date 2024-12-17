import os
from PIL import Image

def check_image_sizes(directory):
    # 遍历指定目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名以确定是否为图片文件
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                try:
                    # 使用PIL打开图片
                    with Image.open(file_path) as img:
                        # 获取图片尺寸
                        width, height = img.size
                        # 检查尺寸是否为512x512
                        if width != 512 or height != 512:
                            print(f"图片大小不符合要求: {file_path} (尺寸: {width}x{height})")
                except Exception as e:
                    print(f"无法打开图片: {file_path} (错误: {e})")

if __name__ == "__main__":
    # 指定要检查的目录路径
    directory_to_check = r"D:\PROJECT\AI-Project\ChangeDetect\SCanNet\datasets\SECOND\ST\val\im1"
    check_image_sizes(directory_to_check)