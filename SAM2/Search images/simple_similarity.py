import os
import glob
from PIL import Image
import imagehash
import cv2
import numpy as np


def calculate_image_hash(img_path):
    """计算图片哈希值"""
    try:
        with Image.open(img_path) as img:
            return imagehash.dhash(img)
    except Exception as e:
        print(f"处理图片 {img_path} 时出错: {e}")
        return None


def calculate_histogram_similarity(img_path1, img_path2):
    """计算直方图相似度"""
    try:
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        
        if img1 is None or img2 is None:
            return 0
        
        # 转换为HSV颜色空间并计算直方图
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, correlation)
        
    except Exception as e:
        print(f"计算直方图相似度时出错: {e}")
        return 0


def calculate_template_matching(img_path1, img_path2):
    """计算模板匹配相似度"""
    try:
        target = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        template = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        
        if target is None or template is None:
            return 0
        
        # 如果模板比目标图片大，交换它们
        if template.shape[0] > target.shape[0] or template.shape[1] > target.shape[1]:
            target, template = template, target
        
        result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        return max(0, max_val)
        
    except Exception as e:
        print(f"计算模板匹配相似度时出错: {e}")
        return 0


def get_all_images(folder_path):
    """获取文件夹中所有图片文件"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(folder_path, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(list(set(image_files)))


def find_similar_images(target_image, search_folder, method='hash', top_n=5):
    """
    找到最相似的图片
    :param target_image: 目标图片路径
    :param search_folder: 搜索文件夹路径
    :param method: 相似度计算方法 ('hash', 'histogram', 'template')
    :param top_n: 返回前N个最相似的结果
    """
    print(f"目标图片: {target_image}")
    print(f"搜索文件夹: {search_folder}")
    print(f"计算方法: {method}")
    
    # 检查文件是否存在
    if not os.path.exists(target_image):
        print(f"目标图片不存在: {target_image}")
        return []
    
    if not os.path.exists(search_folder):
        print(f"搜索文件夹不存在: {search_folder}")
        return []
    
    # 获取所有图片文件
    image_files = get_all_images(search_folder)
    if not image_files:
        print("未找到任何图片文件")
        return []
    
    print(f"找到 {len(image_files)} 张图片，开始计算相似度...")
    
    similarities = []
    
    # 如果使用哈希方法，先计算目标图片的哈希值
    target_hash = None
    if method == 'hash':
        target_hash = calculate_image_hash(target_image)
        if target_hash is None:
            print("无法处理目标图片")
            return []
    
    for i, img_path in enumerate(image_files, 1):
        # 跳过目标图片本身
        if os.path.abspath(img_path) == os.path.abspath(target_image):
            continue
        
        if method == 'hash':
            img_hash = calculate_image_hash(img_path)
            if img_hash is not None and target_hash is not None:
                distance = target_hash - img_hash  # 汉明距离
                similarity_score = max(0, (64 - distance) / 64 * 100)
                similarities.append((img_path, distance, similarity_score))
        
        elif method == 'histogram':
            hist_similarity = calculate_histogram_similarity(target_image, img_path)
            similarity_score = hist_similarity * 100
            distance = 1 - hist_similarity
            similarities.append((img_path, distance, similarity_score))
        
        elif method == 'template':
            template_similarity = calculate_template_matching(target_image, img_path)
            similarity_score = template_similarity * 100
            distance = 1 - template_similarity
            similarities.append((img_path, distance, similarity_score))
        
        # 显示进度
        if i % 20 == 0 or i == len(image_files):
            print(f"进度: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%)")
    
    if not similarities:
        print("没有成功处理的图片")
        return []
    
    # 按距离排序（距离越小越相似）
    similarities.sort(key=lambda x: x[1])
    
    # 显示结果
    print(f"\n{'='*80}")
    print(f"相似度排行榜 (前{top_n}名) - 方法: {method.upper()}")
    print(f"{'='*80}")
    print(f"{'排名':<4} {'距离/差异':<12} {'相似度':<10} {'图片路径'}")
    print("-" * 80)
    
    top_results = similarities[:top_n]
    for i, (img_path, distance, similarity) in enumerate(top_results, 1):
        rel_path = os.path.relpath(img_path)
        if method == 'hash':
            distance_str = f"{int(distance)}"
        else:
            distance_str = f"{distance:.4f}"
        print(f"{i:<4} {distance_str:<12} {similarity:.1f}%{'':<6} {rel_path}")
    
    # 显示最相似的图片
    if top_results:
        best_match = top_results[0]
        print(f"\n最相似的图片:")
        print(f"  路径: {os.path.relpath(best_match[0])}")
        if method == 'hash':
            print(f"  汉明距离: {int(best_match[1])}")
        else:
            print(f"  差异度: {best_match[1]:.4f}")
        print(f"  相似度: {best_match[2]:.1f}%")
    
    return top_results


if __name__ == '__main__':
    # 配置参数
    target_image = 'sample1.jpg'
    search_folder = 'opt_result/2ded08aa754066e9235454e451f5703'
    
    # 选择方法: 'hash', 'histogram', 'template'
    method = 'histogram'  # 可以修改这里选择不同的方法
    
    print(f"开始使用 {method.upper()} 方法进行图像相似度分析...")
    results = find_similar_images(target_image, search_folder, method=method, top_n=10)
    
    if results:
        print(f"\n✅ 分析完成！找到 {len(results)} 个相似的图片")
    else:
        print(f"\n❌ 分析失败或未找到相似图片") 