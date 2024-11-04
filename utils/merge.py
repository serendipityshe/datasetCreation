from PIL import Image
import os

def merge_multiple_images(image_paths, direction, output_path):
    """
    将多张图片按照指定的x,y方向进行拼接

    参数:
        image_paths (list): 图片路径的列表
        direction (str): 拼接方向，'x' 表示水平拼接，'y' 表示垂直拼接
        output_path (str): 输出图片的路径
    """
    images = [Image.open(path) for path in image_paths]

    # 确保所有图片在拼接方向上的尺寸是一致的
    if direction == 'x':
        # 水平拼接，检查高度是否一致
        assert all(img.height == images[0].height for img in images), "所有图片的高度必须一致才能进行水平拼接"
        total_width = sum(img.width for img in images)
        max_height = images[0].height
        new_img = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width

    elif direction == 'y':
        # 垂直拼接，检查宽度是否一致
        assert all(img.width == images[0].width for img in images), "所有图片的宽度必须一致才能进行垂直拼接"
        max_width = images[0].width
        total_height = sum(img.height for img in images)
        new_img = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for img in images:
            new_img.paste(img, (0, y_offset))
            y_offset += img.height

    else:
        raise ValueError("direction 参数必须为 'x' 或 'y'")

    # 保存拼接后的图片
    new_img.save(output_path)

# 使用示例
image_paths = ['/home/hz/下载/4/osgb/DOM/4.tif/4.tif_1_5_DOM.tif', '/home/hz/下载/4/osgb/DOM/4.tif/4.tif_1_6_DOM.tif']  # 图片路径列表
output_path = './images/merged_images.jpg'  # 输出图片路径
merge_multiple_images(image_paths, 'y', output_path)  # 水平拼接多张图片


