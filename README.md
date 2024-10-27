# 将rolableImg工具标注的图像转换可供yolo进行训练的格式

## 1. 数据集准备与矫正

关于roLabelImg标注工具的使用可以参考这个博客： [旋转标注工具rolabelImg使用教程](https://blog.csdn.net/qq_41672428/article/details/107690102)。

标注之后会生成xml格式的标注文件，内容如下所示： 
![alt text](image.png)

其中cx， cy表示目标框的中心点坐标，w, h表示目标框的宽和高。angle表示目标框的旋转角度。

在本项目中的./utils/dataPrePare中提供了： 
- 中文路径转换问题
- 数据比较

## 2. 格式转换

> label文件格式转换的逻辑是：
> - 先将格式转换为obb检测通用数据集DOTA数据集（类似COCO数据集在目标检测的地位）的xml格式，
> - 再转换为txt形式的DOTA数据集格式
> - 再通过yolov8中自带的转换脚本，将DOTA数据集格式转换为yolov8的txt格式。

## 3. 数据增强

