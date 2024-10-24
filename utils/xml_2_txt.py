import xml.etree.ElementTree as ET
import os

name2id = {'daylightingBand': 0,}  # 根据自己数据集的类别进行修改

def convert(img_size, box):
    dw = 1. / img_size[0]
    dh = 1. / img_size[1]
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def decode_xml(xml_folder_path, xml_name):
    txt_name = os.path.join('/home/hz/project/datasetCreation/data/custom/labels', xml_name[0:-4] + '.txt')
    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(txt_name), exist_ok=True)
    txt_file = open(txt_name, 'w')
    
    xml_path = os.path.join(xml_folder_path, xml_name)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    img_w = int(root.find('size').find('width').text)
    img_h = int(root.find('size').find('height').text)
    
    for obj in root.iter('object'):
        label_name = obj.find('name').text
        bbox = obj.find('robndbox')
        if bbox is not None:  # 检查bbox是否为None
            x1 = float(bbox.find('cx').text)
            y1 = float(bbox.find('cy').text)
            x2 = float(bbox.find('w').text)
            y2 = float(bbox.find('h').text)
            
            bb = (x1, y1, x2, y2)
            converted_bbox = convert((img_w, img_h), bb)
            txt_file.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in converted_bbox]) + '\n')
        else:
            print(f"Warning: No bbox found for object with label {label_name}")
    
    txt_file.close()

if __name__ == "__main__":
    xml_folder_path = 'data/daylightingBand/labels/test'
    xml_names = os.listdir(xml_folder_path)
    for xml_name in xml_names:
        if xml_name.endswith('.xml'):  # 确保只处理.xml文件
            decode_xml(xml_folder_path, xml_name)