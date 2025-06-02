# Vision Transformer (ViT) 图像相似度搜索

本项目已新增 Vision Transformer (ViT) 方法用于图像相似度计算。ViT是一种基于Transformer架构的深度学习模型，在图像识别任务中表现优异。

## 🆕 新增功能

### ViT方法特点
- **高精度**: 基于Transformer架构，能够捕捉图像的全局特征
- **语义理解**: 相比传统方法，更好地理解图像内容和语义
- **预训练模型**: 使用在ImageNet-21k上预训练的模型
- **鲁棒性**: 对光照、角度、尺度变化具有较好的鲁棒性

### 支持的方法
现在支持以下图像相似度计算方法：

1. **传统方法**:
   - `hash` - 感知哈希
   - `histogram` - 直方图比较
   - `template` - 模板匹配
   - `ssim` - 结构相似度
   - `sift` - SIFT特征匹配
   - `surf` - SURF特征匹配
   - `orb` - ORB特征匹配

2. **深度学习方法**:
   - `resnet50` - ResNet50特征匹配
   - `vit` - **Vision Transformer特征匹配** ⭐ **新增**
   - `combined` - 综合方法（包含ViT）

## 📦 安装依赖

### 自动安装
```bash
python install_requirements.py
```

### 手动安装
```bash
# 基础依赖
pip install pillow imagehash opencv-python numpy scikit-image

# 深度学习依赖
pip install torch torchvision transformers
```

## 🚀 使用方法

### 1. 测试ViT功能
```bash
python test_vit.py
```

### 2. 使用ViT进行图像搜索
修改 `find_similar_image.py` 中的方法列表：
```python
methods = ["vit"]  # 或者 ["resnet50", "vit", "combined"]
```

### 3. 代码示例
```python
from find_similar_image import calculate_vit_similarity, extract_vit_features

# 计算两张图片的ViT相似度
similarity = calculate_vit_similarity("image1.jpg", "image2.jpg")
print(f"相似度: {similarity:.4f}")

# 提取ViT特征
features = extract_vit_features("image.jpg")
print(f"特征维度: {features.shape}")
```

## ⚙️ 技术细节

### ViT模型配置
- **模型**: `google/vit-base-patch16-224-in21k`
- **输入尺寸**: 224×224像素
- **特征维度**: 768维
- **特征提取**: 使用[CLS] token的输出作为图像表示

### 相似度计算
- **方法**: 余弦相似度
- **归一化**: L2归一化
- **范围**: 0-1（1表示完全相似）

### 性能优化
- **单例模式**: 模型只加载一次，避免重复加载
- **批量处理**: 支持批量特征提取
- **内存管理**: 使用torch.no_grad()减少内存占用

## 📊 性能对比

| 方法 | 精度 | 速度 | 内存占用 | 适用场景 |
|------|------|------|----------|----------|
| Hash | 低 | 极快 | 极低 | 完全相同图片检测 |
| Histogram | 中 | 快 | 低 | 颜色相似图片检测 |
| SIFT | 中高 | 中 | 中 | 物体检测，几何变换鲁棒 |
| ResNet50 | 高 | 中慢 | 高 | 通用图像相似度 |
| **ViT** | **很高** | **中慢** | **高** | **语义相似度，内容理解** |

## 🔧 故障排除

### 常见问题

1. **transformers库安装失败**
   ```bash
   pip install --upgrade pip
   pip install transformers
   ```

2. **模型下载失败**
   - 检查网络连接
   - 使用镜像源：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers`

3. **内存不足**
   - 减少批量处理的图片数量
   - 使用较小的ViT模型

4. **CUDA相关错误**
   - ViT默认使用CPU，如需GPU加速请修改代码

### 调试模式
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 使用建议

### 选择合适的方法
- **快速筛选**: 使用`hash`方法
- **颜色相似**: 使用`histogram`方法  
- **物体检测**: 使用`sift`或`orb`方法
- **语义相似**: 使用`vit`方法 ⭐
- **最佳效果**: 使用`combined`方法

### 参数调优
- 可以调整combined方法中各算法的权重
- ViT在combined方法中默认权重为20%

## 🤝 贡献

欢迎提交Issue和Pull Request来改进ViT功能！

## 📄 许可证

本项目遵循原有许可证。ViT模型来自Hugging Face，遵循Apache 2.0许可证。 