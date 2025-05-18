import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

# 数据增强和预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ConvNeXt 输入尺寸
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
dataset = ImageFolder(root='./images', transform=train_transform)

# 划分训练、验证、测试集
num_total = len(dataset)
indices = list(range(num_total))
np.random.seed(42)  # 固定随机种子
np.random.shuffle(indices)

train_split = int(0.7 * num_total)
val_split = int(0.85 * num_total)

train_indices = indices[:train_split]
val_indices = indices[train_split:val_split]
test_indices = indices[val_split:]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# 创建 DataLoader
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler, num_workers=4)
val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler, num_workers=4)
test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler, num_workers=4)

# 修改数据集变换为验证/测试变换
val_dataset = ImageFolder(root='./images', transform=val_test_transform)
test_dataset = ImageFolder(root='./images', transform=val_test_transform)
val_loader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, sampler=test_sampler, num_workers=4)

# 打印类别信息
print("Classes:", dataset.classes)  # ['flower1', 'flower2', 'flower3']