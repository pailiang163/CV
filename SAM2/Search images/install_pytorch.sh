#!/bin/bash

echo "========================================"
echo "安装PyTorch依赖 - Linux/Mac版本"
echo "========================================"
echo

echo "正在检查Python版本..."
python3 --version
if [ $? -ne 0 ]; then
    echo "错误: 未找到Python3，请先安装Python 3.7+"
    exit 1
fi

echo
echo "正在安装PyTorch和TorchVision..."
echo "这可能需要几分钟时间，请耐心等待..."
echo

pip3 install torch torchvision

if [ $? -ne 0 ]; then
    echo
    echo "安装失败，尝试安装CPU版本..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo
echo "========================================"
echo "测试PyTorch安装"
echo "========================================"

python3 -c "import torch; print('PyTorch版本:', torch.__version__)"
python3 -c "import torchvision; print('TorchVision版本:', torchvision.__version__)"

if [ $? -ne 0 ]; then
    echo "安装验证失败"
    exit 1
fi

echo
echo "✓ PyTorch安装成功！"
echo
echo "现在您可以运行以下命令测试ResNet50功能："
echo "python3 test_resnet50.py"
echo
echo "或者直接运行图像相似度分析："
echo "python3 find_similar_image.py"
echo 