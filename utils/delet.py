import os
import shutil



delete_img = {'26.tif_9_9_DOM000', '20.tif_2_1_DOM200 (副本 2)', '26.tif_8_9_DOM100', '26.tif_17_5_DOM001 (副本 2)', '26.tif_8_9_DOM000', '26.tif_17_5_DOM200 (副本 9)', '26.tif_17_5_DOM011', '26.tif_8_9_DOM000 (副本)', '20.tif_2_1_DOM000', '26.tif_17_5_DOM000 (副本)', '26.tif_8_9_DOM100 (副本 2)', '26.tif_8_9_DOM011', '26.tif_8_9_DOM011 (副本 2)', '26.tif_8_9_DOM001 (副本 2)', '26.tif_9_9_DOM000 (副本)', '26.tif_8_9_DOM001', '26.tif_17_5_DOM200 (副本 6)', '26.tif_17_5_DOM200 (副本 8)', '26.tif_17_5_DOM200', '20.tif_2_1_DOM200 (副本)', '26.tif_9_9_DOM011 (副本)', '26.tif_9_9_DOM200 (副本 2)', '26.tif_17_5_DOM011 (副本 2)', '26.tif_8_9_DOM200', '26.tif_8_9_DOM100 (副本)', '26.tif_8_9_DOM001 (副本)', '26.tif_9_9_DOM200 (副本 5)', '26.tif_17_5_DOM010 (副本 2)', '26.tif_8_9_DOM200 (副本)', '26.tif_17_5_DOM200 (副本)', '26.tif_8_9_DOM000 (副本 2)', '26.tif_17_5_DOM100', '26.tif_17_5_DOM001 (副本)', '26.tif_17_5_DOM011 (副本)', '26.tif_9_9_DOM200 (副本 10)', '26.tif_9_9_DOM010 (副本 2)', '26.tif_17_5_DOM000', '26.tif_17_5_DOM200 (副本 11)', '26.tif_9_9_DOM200 (副本 3)', '26.tif_17_5_DOM200 (副本 7)', '26.tif_9_9_DOM100 (副本)', '20.tif_2_1_DOM200 (副本 3)', '26.tif_9_9_DOM010', '26.tif_9_9_DOM000 (副本 2)', '20.tif_2_1_DOM000 (副本)', '26.tif_17_5_DOM200 (副本 5)', '26.tif_17_5_DOM200 (副本 10)', '26.tif_8_9_DOM010', '26.tif_17_5_DOM010', '26.tif_17_5_DOM001', '26.tif_9_9_DOM100 (副本 2)', '26.tif_17_5_DOM200 (副本 4)', '26.tif_8_9_DOM200 (副本 2)', '26.tif_9_9_DOM200 (副本 4)', '26.tif_8_9_DOM011 (副本)', '26.tif_9_9_DOM100', '26.tif_9_9_DOM200 (副本 11)', '26.tif_17_5_DOM100 (副本)', '26.tif_9_9_DOM011 (副本 2)', '26.tif_9_9_DOM011', '26.tif_17_5_DOM200 (副本 3)', '26.tif_8_9_DOM010 (副本 2)', '26.tif_17_5_DOM200 (副本 2)', '26.tif_17_5_DOM100 (副本 2)', '20.tif_2_1_DOM000 (副本 3)'}


def move_matching_files(source_directory, destination_directory):
    """
    将指定源目录下与 delete_img 集合中文件名（不考虑后缀）相同的文件移动到目标目录。
    
    :param source_directory: 要搜索文件的源目录路径（字符串）
    :param destination_directory: 要将文件移动到的目标目录路径（字符串）
    """
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    # 遍历源目录中的所有文件
    for filename in os.listdir(source_directory):
        # 获取文件的基本名称（不包括扩展名）
        base_name, ext = os.path.splitext(filename)
        
        # 检查基本名称是否在 delete_img 集合中
        if base_name in delete_img:
            # 构建文件的源路径和目标路径
            source_path = os.path.join(source_directory, filename)
            destination_path = os.path.join(destination_directory, filename)
            
            # 如果文件存在，则将其移动到目标目录
            if os.path.isfile(source_path):
                try:
                    shutil.move(source_path, destination_path)
                    print(f"Moved file from {source_path} to {destination_path}")
                except OSError as e:
                    print(f"Error moving file from {source_path} to {destination_path}: {e.strerror}")

# 使用示例：调用函数以移动匹配的文件
source_dir = "/media/hz/新加卷/data/change/images/train"  # 替换为实际的源目录路径
destination_dir = "/media/hz/新加卷/data/change/images/del"  # 替换为实际的目标目录路径
move_matching_files(source_dir, destination_dir)

