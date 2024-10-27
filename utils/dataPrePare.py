import os
import re

class DataPreparer:
    def __init__(self):
        pass

    def get_filenames_without_extension(self, folder_path):
        """
        获取文件夹中所有文件的名称，不包含扩展名。
        """
        filenames = os.listdir(folder_path)
        filenames_without_extension = set(os.path.splitext(file)[0] for file in filenames)
        return filenames_without_extension

    def compare_folders(self, folder_path_1, folder_path_2):
        """
        比较两个文件夹中的文件，返回各自缺失的文件名。
        """
        folder_1_files = self.get_filenames_without_extension(folder_path_1)
        folder_2_files = self.get_filenames_without_extension(folder_path_2)

        missing_in_folder_1 = folder_2_files - folder_1_files
        missing_in_folder_2 = folder_1_files - folder_2_files

        return missing_in_folder_1, missing_in_folder_2

    def replace_chinese_in_filenames(self, directory_path, translation_dict):
        """
        将文件名中的中文字符替换为英文翻译。
        """
        if not os.path.isdir(directory_path):
            print(f"The directory '{directory_path}' does not exist.")
            return

        for filename in os.listdir(directory_path):
            if re.search(r'[\u4e00-\u9fff]+', filename):
                new_filename = filename
                for chinese, english in translation_dict.items():
                    new_filename = new_filename.replace(chinese, english)

                if new_filename != filename:
                    old_file_path = os.path.join(directory_path, filename)
                    new_file_path = os.path.join(directory_path, new_filename)
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed '{filename}' to '{new_filename}'")

# 使用示例
if __name__ == "__main__":
    preparer = DataPreparer()

    # 比较文件夹
    folder_path_1 = './data/xml'
    folder_path_2 = './data/txt'
    missing_in_1, missing_in_2 = preparer.compare_folders(folder_path_1, folder_path_2)
    print(f"Missing in folder 1: {missing_in_1}")
    print(f"Missing in folder 2: {missing_in_2}")

    # # 替换文件名中的中文
    # directory_path = './images'
    # translation_dict = {
    #     "砖石双坡屋顶": "brickHouse",
    #     "彩钢瓦双坡厂房": "caiChangFang",
    #     "测试": "test",
    #     # 添加更多翻译
    # }
    # preparer.replace_chinese_in_filenames(directory_path, translation_dict)