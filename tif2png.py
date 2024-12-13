from PIL import Image

# 将tiff文件转换为png格式
img_path = 'E:/xunlei/TDOM/0802-1028/patches_2048/A/tif1.tif'

with Image.open(img_path) as im:
    im.save('tif1.png', 'PNG')     
