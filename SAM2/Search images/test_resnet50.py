#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet50图像相似度测试脚本
测试PyTorch ResNet50特征提取和相似度计算功能
"""

import os
import sys
from find_similar_image import (
    PYTORCH_AVAILABLE, 
    calculate_resnet50_similarity,
    extract_resnet50_features,
    get_resnet50_model
)

def test_pytorch_installation():
    """测试PyTorch是否正确安装"""
    print("="*60)
    print("测试PyTorch安装状态")
    print("="*60)
    
    if PYTORCH_AVAILABLE:
        print("✓ PyTorch已安装")
        try:
            import torch
            print(f"✓ PyTorch版本: {torch.__version__}")
            
            import torchvision
            print(f"✓ TorchVision版本: {torchvision.__version__}")
            
            # 测试CUDA可用性
            if torch.cuda.is_available():
                print(f"✓ CUDA可用，设备数量: {torch.cuda.device_count()}")
                print(f"✓ 当前CUDA设备: {torch.cuda.get_device_name()}")
            else:
                print("⚠ CUDA不可用，将使用CPU")
            
            return True
        except Exception as e:
            print(f"✗ PyTorch导入错误: {e}")
            return False
    else:
        print("✗ PyTorch未安装")
        print("请运行: pip install torch torchvision")
        return False

def test_resnet50_model():
    """测试ResNet50模型加载"""
    print("\n" + "="*60)
    print("测试ResNet50模型加载")
    print("="*60)
    
    try:
        model, transform = get_resnet50_model()
        print("✓ ResNet50模型加载成功")
        print(f"✓ 模型类型: {type(model)}")
        print(f"✓ 预处理变换: {type(transform)}")
        return True
    except Exception as e:
        print(f"✗ ResNet50模型加载失败: {e}")
        return False

def test_feature_extraction():
    """测试特征提取功能"""
    print("\n" + "="*60)
    print("测试特征提取功能")
    print("="*60)
    
    # 查找测试图片
    test_images = []
    for img_name in ['sample1.jpg', 'sample2.jpg']:
        if os.path.exists(img_name):
            test_images.append(img_name)
    
    if not test_images:
        print("✗ 未找到测试图片 (sample1.jpg 或 sample2.jpg)")
        return False
    
    print(f"找到测试图片: {test_images}")
    
    for img_path in test_images:
        try:
            print(f"\n测试图片: {img_path}")
            features = extract_resnet50_features(img_path)
            
            if features is not None:
                print(f"✓ 特征提取成功")
                print(f"✓ 特征维度: {features.shape}")
                print(f"✓ 特征范围: [{features.min():.4f}, {features.max():.4f}]")
                print(f"✓ 特征L2范数: {(features**2).sum()**0.5:.4f}")
            else:
                print(f"✗ 特征提取失败")
                return False
                
        except Exception as e:
            print(f"✗ 特征提取出错: {e}")
            return False
    
    return True

def test_similarity_calculation():
    """测试相似度计算功能"""
    print("\n" + "="*60)
    print("测试相似度计算功能")
    print("="*60)
    
    # 查找测试图片
    test_images = []
    for img_name in ['sample1.jpg', 'sample2.jpg']:
        if os.path.exists(img_name):
            test_images.append(img_name)
    
    if len(test_images) < 2:
        print("✗ 需要至少2张测试图片进行相似度计算")
        return False
    
    try:
        img1, img2 = test_images[0], test_images[1]
        print(f"计算相似度: {img1} vs {img2}")
        
        similarity = calculate_resnet50_similarity(img1, img2)
        
        print(f"✓ 相似度计算成功")
        print(f"✓ 相似度分数: {similarity:.4f}")
        print(f"✓ 相似度百分比: {similarity*100:.1f}%")
        
        # 测试自相似度（应该接近1.0）
        self_similarity = calculate_resnet50_similarity(img1, img1)
        print(f"✓ 自相似度: {self_similarity:.4f} (应该接近1.0)")
        
        if self_similarity > 0.99:
            print("✓ 自相似度测试通过")
        else:
            print("⚠ 自相似度偏低，可能存在问题")
        
        return True
        
    except Exception as e:
        print(f"✗ 相似度计算出错: {e}")
        return False

def main():
    """主测试函数"""
    print("ResNet50图像相似度功能测试")
    print("="*60)
    
    # 测试步骤
    tests = [
        ("PyTorch安装", test_pytorch_installation),
        ("ResNet50模型", test_resnet50_model),
        ("特征提取", test_feature_extraction),
        ("相似度计算", test_similarity_calculation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            
            if not result:
                print(f"\n⚠ {test_name}测试失败，跳过后续测试")
                break
                
        except Exception as e:
            print(f"\n✗ {test_name}测试出现异常: {e}")
            results.append((test_name, False))
            break
    
    # 显示测试总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 项测试通过")
    
    if passed == len(results):
        print("\n🎉 所有测试通过！ResNet50功能可以正常使用")
        print("\n您现在可以在find_similar_image.py中使用 'resnet50' 方法")
        print("示例: python find_similar_image.py")
    else:
        print("\n❌ 部分测试失败，请检查PyTorch安装或图片文件")
        print("\n安装PyTorch: pip install torch torchvision")

if __name__ == '__main__':
    main() 