

from PIL import Image
import numpy as np

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

# 定义颜色映射表
# ST_COLORMAP = [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0], [255, 0, 0]]
ST_COLORMAP = [[255, 255, 255], [0, 128, 0], [128, 128, 128], [0, 255, 0], [0, 0, 255], [128, 0, 0], [255, 0, 0]]

# 读取图片
image_path = r'D:\PROJECT\AI-Project\ChangeDetect\MambaCD\data\SECOND\train\GT_T1\00013.png' # 替换为你的图片路径
image = Image.open(image_path)

# 将灰度图转换为NumPy数组
image_array = np.array(image)
img = Index2Color(image_array)
print(img)

# 将img以RGB格式保存为png图片
img = Image.fromarray(np.uint8(img), mode="RGB")
img.save("test.png", format="PNG")



