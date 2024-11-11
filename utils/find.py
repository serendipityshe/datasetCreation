import os
from pyhocon import ConfigFactory


def get_filenames_without_extension(directory):
    """
    获取指定目录下所有文件的名称，不包括扩展名。
    
    :param directory: 要搜索文件的目录路径（字符串）
    :return: 不包含扩展名的文件名列表（列表）
    """
    filenames = []  # 初始化一个空列表来存储文件名
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查是否为文件而不是目录
        if os.path.isfile(os.path.join(directory, filename)):
            # 获取文件的基本名称（不包括扩展名）
            base_name, _ = os.path.splitext(filename)
            # 将基本名称添加到列表中
            filenames.append(base_name)
    
    # 返回文件名列表
    return filenames

# 使用示例：调用函数并打印结果

config = ConfigFactory.parse_file('conf/config.conf')
path = config.get('file_path.path')


directory_path = path # 替换为实际的目录路径
file_names = get_filenames_without_extension(directory_path)
print(file_names)
