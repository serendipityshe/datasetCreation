import os 

class dataPre:
    '''数据预处理类：
    1. 获取该路径下所有文件名称
    2. 返回缺失的文件名称
    3. 删除缺失的文件
    '''
    def __init__(self, folder_path, save_path) -> None:
        self.folder_path = folder_path
        self.save_path = save_path
    
    def get_filename_without_extension(self, folder_path):
        file_list = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if not file.endswith('.txt'):
                    continue
                filename = os.path.splitext(file)[0]
                file_list.append(filename)


    def replace_ch2en(self):
        pass

    def compile_folder(self):
        pass

    def delete_missing_file(self):
        pass


