import os
import re
import argparse

def replace_chinese_in_filenames(directory_path: str, translation_dict: dict):
    """
    Replace Chinese characters in filenames with their English translations.

    :param directory_path: Path to the directory containing the files.
    :param translation_dict: Dictionary mapping Chinese characters or phrases to their English translations.
    """
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        print(f"The directory '{directory_path}' does not exist.")
        return

    # Iterate over files in the directory
    for filename in os.listdir(directory_path):
        # Check if the filename contains Chinese characters
        if re.search(r'[\u4e00-\u9fff]+', filename):
            # Replace Chinese characters with their English translations
            new_filename = filename
            for chinese, english in translation_dict.items():
                new_filename = new_filename.replace(chinese, english)

            # If the filename has changed, rename the file
            if new_filename != filename:
                old_file_path = os.path.join(directory_path, filename)
                new_file_path = os.path.join(directory_path, new_filename)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{filename}' to '{new_filename}'")

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default=r'E:\datasets\train\img', help='directory path')

# Example usage
if __name__ == "__main__":
    # Define the directory path and translation dictionary
    translation_dict = {
        "砖石双坡屋顶": "brickHouse",
        "彩钢瓦双坡厂房": "caiChangFang",
        "南校区三维模型": "southCampus",
        "厂房双被": "changFang"
        # Add more translations as needed
    }

    args = parser.parse_args()
    # Call the function to replace Chinese characters in filenames
    replace_chinese_in_filenames(args.dir, translation_dict)