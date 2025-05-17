import os
import shutil
import random

Images_path = './VOCdevkit/VOC2007/JPEGImages'  # 源图路径
Labels_path = './VOCdevkit/VOC2007/labels'  # 源标签路径

train_labels = './VOCdevkit/VOC2007/labels/train2017'  # train标签路径
val_labels = './VOCdevkit/VOC2007/labels/val2017'  # val标签路径

train_images = './VOCdevkit/VOC2007/images/train2017'  # 保存train图像路径
val_images = './VOCdevkit/VOC2007/images/val2017'  # 保存val图像路径

radio = 0.2  # 按照比例划分的验证集比例
nums = 100  # 按照数量划分的验证集数量
is_radio = False  # 如果为True，则按照比例进行划分，否则按照数量划分

# 判断文件夹是否存在，不存在即创建
if not os.path.exists(train_images):
    os.mkdir(train_images)
if not os.path.exists(val_images):
    os.mkdir(val_images)

if not os.path.exists(train_labels):
    os.mkdir(train_labels)
if not os.path.exists(val_labels):
    os.mkdir(val_labels)


Imgs = os.listdir(Images_path)
if is_radio:
    val_nums = int(len(Imgs) * radio)
else:
    val_nums = nums
val_Imgs = random.sample(Imgs, val_nums)

for val_name in val_Imgs:
    shutil.move(os.path.join(Images_path, val_name), os.path.join(val_images, val_name))

    val_name = val_name[:-3] + 'txt'  # jpg2txt
    shutil.move(os.path.join(Labels_path, val_name), os.path.join(val_labels, val_name))

if (len(Imgs) - len(val_Imgs)) > 0:
    for i in val_Imgs:
        if i in Imgs:
            Imgs.remove(i)
    train_Imgs = Imgs
    for train_name in train_Imgs:
        shutil.move(os.path.join(Images_path, train_name), os.path.join(train_images, train_name))

        train_name = train_name[:-3] + 'txt'  # jpg2txt
        shutil.move(os.path.join(Labels_path, train_name), os.path.join(train_labels, train_name))
