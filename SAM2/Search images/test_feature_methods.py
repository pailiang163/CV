#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试特征匹配算法 (SIFT, SURF, ORB)
用于验证新添加的特征匹配功能是否正常工作
"""

import os
import sys
from find_similar_image import (
    calculate_sift_similarity,
    calculate_surf_similarity, 
    calculate_orb_similarity
)


def test_feature_methods():
    """测试所有特征匹配方法"""
    
    # 测试图片路径
    target_image = 'sample1.jpg'
    test_image = 'sample2.jpg'
    
    # 检查测试图片是否存在
    if not os.path.exists(target_image):
        print(f"❌ 目标图片不存在: {target_image}")
        return False
    
    if not os.path.exists(test_image):
        print(f"❌ 测试图片不存在: {test_image}")
        return False
    
    print("🔍 开始测试特征匹配算法...")
    print(f"📷 目标图片: {target_image}")
    print(f"📷 测试图片: {test_image}")
    print("-" * 60)
    
    # 测试SIFT
    print("🔬 测试SIFT特征匹配...")
    try:
        sift_score = calculate_sift_similarity(target_image, test_image)
        print(f"✅ SIFT相似度: {sift_score:.4f} ({sift_score*100:.1f}%)")
    except Exception as e:
        print(f"❌ SIFT测试失败: {e}")
    
    print()
    
    # 测试SURF
    print("🔬 测试SURF特征匹配...")
    try:
        surf_score = calculate_surf_similarity(target_image, test_image)
        print(f"✅ SURF相似度: {surf_score:.4f} ({surf_score*100:.1f}%)")
    except Exception as e:
        print(f"❌ SURF测试失败: {e}")
    
    print()
    
    # 测试ORB
    print("🔬 测试ORB特征匹配...")
    try:
        orb_score = calculate_orb_similarity(target_image, test_image)
        print(f"✅ ORB相似度: {orb_score:.4f} ({orb_score*100:.1f}%)")
    except Exception as e:
        print(f"❌ ORB测试失败: {e}")
    
    print("-" * 60)
    print("🎉 特征匹配算法测试完成！")
    
    return True


def test_opencv_features():
    """测试OpenCV特征检测器是否可用"""
    
    print("🔧 检查OpenCV特征检测器可用性...")
    print("-" * 60)
    
    import cv2
    
    # 测试SIFT
    try:
        sift = cv2.SIFT_create()
        print("✅ SIFT检测器可用")
    except Exception as e:
        print(f"❌ SIFT检测器不可用: {e}")
    
    # 测试SURF
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
        print("✅ SURF检测器可用 (需要opencv-contrib-python)")
    except AttributeError:
        print("⚠️  SURF检测器不可用 (需要安装opencv-contrib-python)")
    except Exception as e:
        print(f"❌ SURF检测器错误: {e}")
    
    # 测试ORB
    try:
        orb = cv2.ORB_create()
        print("✅ ORB检测器可用")
    except Exception as e:
        print(f"❌ ORB检测器不可用: {e}")
    
    print("-" * 60)


if __name__ == '__main__':
    print("🚀 特征匹配算法测试程序")
    print("=" * 60)
    
    # 检查OpenCV特征检测器
    test_opencv_features()
    print()
    
    # 测试特征匹配功能
    test_feature_methods() 