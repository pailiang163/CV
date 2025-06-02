#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViT功能测试脚本
"""

import os
from find_similar_image import VIT_AVAILABLE, extract_vit_features, calculate_vit_similarity

def test_vit_functionality():
    """测试ViT功能"""
    print("="*60)
    print("ViT (Vision Transformer) 功能测试")
    print("="*60)
    
    # 检查ViT是否可用
    print(f"ViT可用性: {VIT_AVAILABLE}")
    
    if not VIT_AVAILABLE:
        print("❌ ViT不可用，请安装transformers库:")
        print("   pip install transformers")
        return False
    
    # 检查是否有测试图片
    test_image = "sample1.jpg"
    if not os.path.exists(test_image):
        print(f"❌ 测试图片 {test_image} 不存在")
        print("请确保当前目录下有测试图片")
        return False
    
    print(f"✅ 找到测试图片: {test_image}")
    
    try:
        # 测试特征提取
        print("\n正在测试ViT特征提取...")
        features = extract_vit_features(test_image)
        
        if features is not None:
            print(f"✅ 特征提取成功!")
            print(f"   特征维度: {features.shape}")
            print(f"   特征范围: [{features.min():.4f}, {features.max():.4f}]")
            print(f"   特征均值: {features.mean():.4f}")
            print(f"   特征标准差: {features.std():.4f}")
        else:
            print("❌ 特征提取失败")
            return False
        
        # 测试相似度计算（自己和自己比较，应该接近1）
        print("\n正在测试ViT相似度计算...")
        similarity = calculate_vit_similarity(test_image, test_image)
        print(f"✅ 自相似度: {similarity:.4f}")
        
        if similarity > 0.95:
            print("✅ 相似度计算正常（自相似度接近1）")
        else:
            print("⚠️  相似度计算可能有问题（自相似度应该接近1）")
        
        print("\n🎉 ViT功能测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return False

if __name__ == "__main__":
    success = test_vit_functionality()
    if success:
        print("\n✅ 所有测试通过，ViT功能正常!")
    else:
        print("\n❌ 测试失败，请检查配置和依赖") 