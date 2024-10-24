import xml.etree.ElementTree as ET
import os

def convert_xml_to_txt(xml_folder_path, txt_folder_path):
    # 确保 txt 文件夹存在，如果不存在则创建
    if not os.path.exists(txt_folder_path):
        os.makedirs(txt_folder_path)

    # 遍历 xml 文件夹中的所有文件
    for filename in os.listdir(xml_folder_path):
        if filename.endswith('.xml'):  # 检查文件扩展名是否为 .xml
            xml_file_path = os.path.join(xml_folder_path, filename)
            txt_file_name = os.path.splitext(filename)[0] + '.txt'  # 替换扩展名为 .txt
            txt_file_path = os.path.join(txt_folder_path, txt_file_name)
            
            # 解析 XML 文件
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            # 打开对应的 TXT 文件以写入
            with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                # 遍历 XML 的每个元素，并写入 TXT 文件
                for elem in root.iter():
                    if elem.text:  # 确保 elem.text 不为 None，以避免写入空行
                        txt_file.write(f"{elem.tag}: {elem.text.strip()}\n")
                    for attr in elem.attrib:
                        txt_file.write(f"  {attr}: {elem.attrib[attr]}\n")

# 使用示例
xml_folder_path = 'data/daylightingBand/labels/train'  # 替换为你的 XML 文件夹路径
txt_folder_path = 'data/daylightingBand/labels/train1'  # 替换为你想要存放 TXT 文件的文件夹路径
convert_xml_to_txt(xml_folder_path, txt_folder_path)