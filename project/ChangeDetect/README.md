# 变化检测(Change Detection)

- 数据集准备
1. 自制数据集（标注， 处理标签， mask， 整理）
2. 配置模型进行训练（open-cd）
3. 测试， 推理（大型遥感图片需要裁剪， 融合）。


[labelme json批量转换为png](https://blog.csdn.net/qq_42930154/article/details/123121779)

这里推荐一个更好的标注工具，x_anylabeling，标注好之后，它可以导出为指定格式。在语义变化检测中，我们需要将label_rgb的为深度为24的图片转换成位深度为8的灰度图[rgb2gray.py](./rgb2gray.py)，方便后续处理。

在标注过程中，我们提供整理图片脚本，将多个路径下的图片整合到一个文件夹下，方便后续处理。

同样的为了验证我们的转换结果，我们提供了可视化脚本[checkphoto.py](./checkphoto.py)用于查看转换后的图片

当我们整合好数据集之后，我们提供了数据集的划分脚本[split_data.py](./split_data.py)

使用的是open-cd的集成网络模型， SCANNet， Changeformer。在open-cd中，语义分割的预测脚本有问题，暂不知道如何解决。

本篇将从数据集构建到最后模型的训练，推理。