#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安装图像相似度搜索所需的依赖库
"""

import subprocess
import sys

def install_package(package):
    """安装Python包"""
    try:
        print(f"正在安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package} 安装失败: {e}")
        return False

def main():
    """主函数"""
    print("="*60)
    print("图像相似度搜索依赖安装脚本")
    print("="*60)
    
    # 基础依赖
    basic_packages = [
        "pillow",           # 图像处理
        "imagehash",        # 图像哈希
        "opencv-python",    # OpenCV
        "numpy",            # 数值计算
        "scikit-image",     # 图像处理（SSIM）
    ]
    
    # PyTorch相关（用于ResNet50和ViT）
    pytorch_packages = [
        "torch",
        "torchvision",
        "transformers",     # Hugging Face transformers（用于ViT）
    ]
    
    print("正在安装基础依赖...")
    basic_success = True
    for package in basic_packages:
        if not install_package(package):
            basic_success = False
    
    print("\n正在安装PyTorch和深度学习相关依赖...")
    pytorch_success = True
    for package in pytorch_packages:
        if not install_package(package):
            pytorch_success = False
    
    print("\n" + "="*60)
    print("安装结果总结:")
    print("="*60)
    
    if basic_success:
        print("✅ 基础功能依赖安装成功")
        print("   支持的方法: hash, histogram, template, ssim, sift, surf, orb")
    else:
        print("❌ 基础功能依赖安装失败")
    
    if pytorch_success:
        print("✅ 深度学习依赖安装成功")
        print("   支持的方法: resnet50, vit, combined")
    else:
        print("❌ 深度学习依赖安装失败")
        print("   ResNet50和ViT方法将不可用")
    
    if basic_success and pytorch_success:
        print("\n🎉 所有依赖安装完成！可以使用所有功能。")
    elif basic_success:
        print("\n⚠️  基础功能可用，但深度学习功能不可用。")
    else:
        print("\n❌ 安装失败，请检查网络连接和Python环境。")
    
    print("\n使用说明:")
    print("1. 运行 python test_vit.py 测试ViT功能")
    print("2. 运行 python find_similar_image.py 进行图像相似度搜索")

if __name__ == "__main__":
    main() 