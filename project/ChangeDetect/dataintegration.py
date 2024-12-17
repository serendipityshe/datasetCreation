import os
import shutil

def collect_images(source_dirs):
    images = []
    for dir_path in source_dirs:
        if not os.path.isdir(dir_path):
            print(f"Warning: {dir_path} is not a valid directory.")
            continue
        # 遍历指定路径下的文件，而非子目录
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif')):
                images.append(file_path)
    return images

def rename_and_copy_images(images, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    counter = 1
    for img_path in images:
        base_name, ext = os.path.splitext(os.path.basename(img_path))
        new_name = f"{counter:04d}{'.png'}"  # 使用4位数字命名，不足部分用0填充
        new_path = os.path.join(dest_dir, new_name)
        
        # 检查目标文件是否已存在，若存在则递增计数器直到找到未使用的文件名
        while os.path.exists(new_path):
            counter += 1
            new_name = f"{counter:04d}{'.png'}"
            new_path = os.path.join(dest_dir, new_name)
        
        shutil.copy(img_path, new_path)
        counter += 1

if __name__ == "__main__":
    source_dirs = [
        r"D:\PROJECT\AI-Project\ChangeDetect\MambaCD\data\SECOND\train\GT_T2",
        r"E:\xunlei\EIP-SCD512_EN\label2_2",
        r"D:\PROJECT\AI-Project\ChangeDetect\MambaCD\data\SECOND\test\GT_T2"
        # 可添加更多源目录路径...
    ]
    destination_dir = r"D:\PROJECT\AI-Project\ChangeDetect\SCanNet\datasets\SECOND\label2"
    
    images = collect_images(source_dirs)
    rename_and_copy_images(images, destination_dir)
    print("图片重命名与复制完成。")