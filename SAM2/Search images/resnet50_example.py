#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet50图像相似度使用示例
演示如何使用ResNet50深度学习特征进行图像相似度比较
"""

import os
from find_similar_image import (
    PYTORCH_AVAILABLE,
    find_most_similar_image_by_folders,
    calculate_resnet50_similarity
)

def main():
    print("="*80)
    print("ResNet50深度学习图像相似度分析示例")
    print("="*80)
    
    # 检查PyTorch是否可用
    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch未安装，无法使用ResNet50功能")
        print("请运行以下命令安装PyTorch:")
        print("  Windows: install_pytorch.bat")
        print("  Linux/Mac: ./install_pytorch.sh")
        print("  或手动安装: pip install torch torchvision")
        return
    
    print("✅ PyTorch已安装，可以使用ResNet50功能")
    print()
    
    # 配置参数
    target_image = 'sample1.jpg'
    base_folder = 'opt_result'
    
    # 检查文件是否存在
    if not os.path.exists(target_image):
        print(f"❌ 目标图片不存在: {target_image}")
        print("请确保当前目录下有sample1.jpg文件")
        return
    
    if not os.path.exists(base_folder):
        print(f"❌ 搜索文件夹不存在: {base_folder}")
        print("请确保当前目录下有opt_result文件夹")
        return
    
    print(f"🎯 目标图片: {target_image}")
    print(f"📁 搜索文件夹: {base_folder}")
    print()
    
    # 使用ResNet50方法进行相似度分析
    print("🧠 使用ResNet50深度学习特征进行图像相似度分析...")
    print("注意: 首次运行会下载预训练模型，可能需要几分钟时间")
    print()
    
    try:
        # 对每个子文件夹分别计算相似度
        result = find_most_similar_image_by_folders(
            target_image, 
            base_folder, 
            method='resnet50'
        )
        
        if result:
            print(f"\n🎉 ResNet50分析完成！在 {len(result)} 个文件夹中找到了相似图片")
            
            # 显示最佳匹配的详细信息
            best_folder, best_img, best_distance, best_similarity = result[0]
            print(f"\n📊 ResNet50深度学习分析结果:")
            print(f"   最佳匹配文件夹: {os.path.basename(best_folder)}")
            print(f"   最相似图片: {os.path.basename(best_img)}")
            print(f"   语义相似度: {best_similarity:.1f}%")
            print(f"   特征距离: {best_distance:.4f}")
            
            # 解释结果
            if best_similarity >= 80:
                print("   🔥 语义内容非常相似！")
            elif best_similarity >= 60:
                print("   ✅ 语义内容比较相似")
            elif best_similarity >= 40:
                print("   ⚠️ 语义内容有一定相似性")
            else:
                print("   ❌ 语义内容相似度较低")
                
        else:
            print("❌ ResNet50分析失败，未找到相似图片")
            
    except Exception as e:
        print(f"❌ ResNet50分析过程中出错: {e}")
        print("请检查:")
        print("1. PyTorch是否正确安装")
        print("2. 网络连接是否正常（首次需要下载模型）")
        print("3. 图片文件是否可以正常读取")

def demo_single_comparison():
    """演示单张图片对比"""
    print("\n" + "="*60)
    print("ResNet50单张图片相似度对比演示")
    print("="*60)
    
    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch未安装，跳过演示")
        return
    
    # 查找测试图片
    test_images = []
    for img_name in ['sample1.jpg', 'sample2.jpg']:
        if os.path.exists(img_name):
            test_images.append(img_name)
    
    if len(test_images) < 2:
        print("❌ 需要至少2张测试图片 (sample1.jpg, sample2.jpg)")
        return
    
    img1, img2 = test_images[0], test_images[1]
    print(f"🔍 比较图片: {img1} vs {img2}")
    
    try:
        similarity = calculate_resnet50_similarity(img1, img2)
        print(f"🧠 ResNet50语义相似度: {similarity:.4f} ({similarity*100:.1f}%)")
        
        if similarity > 0.8:
            print("🔥 两张图片在语义内容上非常相似！")
        elif similarity > 0.6:
            print("✅ 两张图片在语义内容上比较相似")
        elif similarity > 0.4:
            print("⚠️ 两张图片在语义内容上有一定相似性")
        else:
            print("❌ 两张图片在语义内容上差异较大")
            
    except Exception as e:
        print(f"❌ 相似度计算失败: {e}")

if __name__ == '__main__':
    main()
    demo_single_comparison()
    
    print("\n" + "="*80)
    print("💡 提示:")
    print("- ResNet50基于深度学习，能理解图像的语义内容")
    print("- 即使外观差异较大，相同物体/场景也能被识别为相似")
    print("- 首次运行会下载预训练模型（约100MB）")
    print("- 后续运行速度会明显提升")
    print("- 如需GPU加速，请安装CUDA版本的PyTorch")
    print("="*80) 