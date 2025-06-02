# SAM2 批量图像分割工具

这个项目提供了使用SAM2（Segment Anything Model 2）对图像进行批量分割的工具。

## 功能特性

- 🚀 **自动批量分割**: 自动处理`images`文件夹中的所有图像
- 📁 **结构化输出**: 每张图像的分割结果保存在以图像名称命名的文件夹中
- 🎯 **多种分割模式**: 支持点击式分割和全自动分割
- 📊 **可视化结果**: 生成掩码、覆盖图和统计图表
- 🔧 **自动下载模型**: 自动下载SAM2权重文件

## 文件结构

```
SAM2/
├── images/                          # 输入图像文件夹
│   ├── 2e0662eae9d27ce35e158dd17fe8f5f.jpg
│   ├── 2ded08aa754066e9235454e451f5703.jpg
│   └── 094ce015646d45ba952ef3d6a0d35a7.jpg
├── output/                          # 输出文件夹（自动创建）
│   ├── 2e0662eae9d27ce35e158dd17fe8f5f/
│   ├── 2ded08aa754066e9235454e451f5703/
│   └── 094ce015646d45ba952ef3d6a0d35a7/
├── requirements.txt                 # 依赖文件
├── auto_segment.py                  # 自动分割脚本（推荐）
├── batch_segment.py                 # 批量点击式分割脚本
└── README.md                        # 说明文档
```

## 安装依赖

1. 首先安装Python依赖包：

```bash
pip install -r requirements.txt
```

2. 如果遇到SAM2安装问题，请手动安装：

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

## 使用方法

### 方法一：自动分割（推荐）

使用`auto_segment.py`进行全自动分割，无需手动指定分割点：

```bash
python auto_segment.py
```

**特点：**
- 全自动分割，无需用户干预
- 使用SAM2的自动掩码生成器
- 自动识别图像中的所有对象
- 生成高质量的分割掩码

### 方法二：网格点分割

使用`batch_segment.py`进行基于网格点的分割：

```bash
python batch_segment.py
```

**特点：**
- 在图像上创建网格点进行分割
- 可以获得更精确的分割结果
- 适合需要特定分割点的场景

## 输出文件说明

对于每张输入图像，脚本会在以图像名称命名的文件夹中生成以下文件：

### 自动分割模式输出：
- `{图像名}_original.jpg` - 原始图像副本
- `{图像名}_overview.png` - 包含所有掩码的概览图
- `{图像名}_mask_001_area_{面积}.png` - 分割掩码（按面积大小排序）
- `{图像名}_overlay_001.jpg` - 掩码覆盖在原图上的效果图

### 网格点分割模式输出：
- `{图像名}_original.jpg` - 原始图像副本
- `{图像名}_mask_{序号}_score_{得分}.png` - 分割结果可视化
- `{图像名}_mask_{序号}.png` - 纯掩码图像

## 配置选项

### 自动分割参数调整

在`auto_segment.py`中可以调整以下参数：

```python
self.mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=32,              # 每边的点数（增加可提高精度但降低速度）
    pred_iou_thresh=0.7,             # IoU阈值（0.0-1.0）
    stability_score_thresh=0.95,     # 稳定性得分阈值
    crop_n_layers=1,                 # 裁剪层数
    crop_n_points_downscale_factor=2, # 裁剪点缩放因子
    min_mask_region_area=100,        # 最小掩码区域面积
)
```

### 模型选择

可以使用不同大小的SAM2模型：

- `sam2_hiera_tiny.yaml` - 最小模型，速度最快
- `sam2_hiera_small.yaml` - 小型模型
- `sam2_hiera_base_plus.yaml` - 基础增强模型
- `sam2_hiera_large.yaml` - 大型模型（默认，精度最高）

## 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA支持（推荐，用于GPU加速）
- 至少8GB内存
- 足够的磁盘空间存储模型权重文件（约2GB）

## 常见问题

### 1. 模型下载失败
如果自动下载模型失败，请手动下载：
1. 创建`checkpoints`文件夹
2. 从[SAM2官方页面](https://github.com/facebookresearch/segment-anything-2)下载对应的权重文件
3. 将权重文件放在`checkpoints/`文件夹中

### 2. 内存不足
如果遇到内存不足的问题：
- 使用更小的模型（如`sam2_hiera_tiny.yaml`）
- 减少`points_per_side`参数
- 一次处理较少的图像

### 3. 分割质量不理想
- 调整`pred_iou_thresh`和`stability_score_thresh`参数
- 增加`points_per_side`以获得更精细的分割
- 尝试不同的模型配置

## 许可证

本项目基于SAM2的许可证条款。请参考[SAM2官方仓库](https://github.com/facebookresearch/segment-anything-2)了解详细的许可证信息。

## 致谢

- Meta AI 团队开发的 [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/segment-anything-2)
- PyTorch 和相关开源库的开发者们 
