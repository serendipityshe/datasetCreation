import os
import re
import random
import shutil
import cv2
from tqdm import tqdm

class DataPrepare:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def replace_chinese_in_filenames(self, translation_dict):
        """
        Replace Chinese characters in filenames with their English translations.
        """
        # Check if the directory exists
        if not os.path.isdir(self.input_dir):
            print(f"The directory '{self.input_dir}' does not exist.")
            return

        # Iterate over files in the directory
        for filename in os.listdir(self.input_dir):
            # Check if the filename contains Chinese characters
            if re.search(r'[\u4e00-\u9fff]+', filename):
                # Replace Chinese characters with their English translations
                new_filename = filename
                for chinese, english in translation_dict.items():
                    new_filename = new_filename.replace(chinese, english)

                # If the filename has changed, rename the file
                if new_filename != filename:
                    old_file_path = os.path.join(self.input_dir, filename)
                    new_file_path = os.path.join(self.input_dir, new_filename)
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed '{filename}' to '{new_filename}'")

    def compare_folders(self, folder_path_1, folder_path_2):
        """
        Compare two folders and return the missing files.
        """
        folder_1_files = set(os.path.splitext(file)[0] for file in os.listdir(folder_path_1))
        folder_2_files = set(os.path.splitext(file)[0] for file in os.listdir(folder_path_2))

        missing_in_folder_1 = folder_2_files - folder_1_files
        missing_in_folder_2 = folder_1_files - folder_2_files

        return missing_in_folder_1, missing_in_folder_2

    def image_data_enhancement(self, ratio=1.0):
        """
        Apply image data enhancement techniques.
        """
        # ... [Add methods for data enhancement like AddWeather, MirrorHorizon, etc.]

    def __lable2txt(self, lableInfo, txtPath):
        """
        Convert label information to text file.
        """
        with open(txtPath, 'w') as f:
            f.writelines([line + os.linesep for line in lableInfo])

    def save_file(self, image_path, label_path, file_name):
        """
        Save image and corresponding label file.
        """
        shutil.copy(image_path, self.output_dir + '/' + file_name)
        self.__lable2txt(label_path, self.output_dir + '/' + file_name + '.txt')