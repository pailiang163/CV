# 实现xml格式转yolov5格式
import xml.etree.ElementTree as ET
import os

# box [xmin,ymin,xmax,ymax]
def convert(size, box):
    x_center = (box[2] + box[0]) / 2.0
    y_center = (box[3] + box[1]) / 2.0
    # 归一化
    x = x_center / size[0]
    y = y_center / size[1]
    # 求宽高并归一化
    w = (box[2] - box[0]) / size[0]
    h = (box[3] - box[1]) / size[1]
    return (x, y, w, h)

# xml2yolo
def convert_annotation(xml_paths, yolo_paths, classes):
    xml_files = os.listdir(xml_paths)
    # 生成无序文件列表
    for file in xml_files:
        
        xml_file_path = os.path.join(xml_paths, file)
        yolo_txt_path = os.path.join(yolo_paths, file.split(".")[0]
                                     + ".txt")
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        size = root.find("size")
        # 获取xml的width和height的值
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        # object标签可能会存在多个，所以要迭代
        with open(yolo_txt_path, 'w') as f:
            for obj in root.iter("object"):
                difficult = obj.find("difficult").text
                # 类别
                cls = obj.find("name").text
                if cls not in classes or difficult == 1:
                    continue
                # 转换成训练模式读取的标签
                cls_id = classes.index(cls)
                xml_box = obj.find("bndbox")
                box = (float(xml_box.find("xmin").text), float(xml_box.find("ymin").text),
                       float(xml_box.find("xmax").text), float(xml_box.find("ymax").text))
                boxex = convert((w, h), box)
                # yolo标准格式类别 x_center,y_center,width,height
                f.write(str(cls_id) + " " + " ".join([str(s) for s in boxex]) + '\n')
        print(f'xml_file :{file} --> txt Saved!')


if __name__ == "__main__":
    # PCB数据的类别
    classes_train = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']   # 修改1，类别
    
    # xml存储地址
    xml_dir = "./VOCdevkit/VOC2007/Annotations/"  # 修改2，读取位置
    
    # yolo存储地址
    yolo_txt_dir = "./VOCdevkit/VOC2007/labels/"   # 修改3，保存位置
    
    # voc转yolo
    convert_annotation(xml_paths=xml_dir, yolo_paths=yolo_txt_dir,
                       classes=classes_train)
