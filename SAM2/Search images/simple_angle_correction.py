import os
import glob
import cv2
import numpy as np
from PIL import Image
import imagehash
import matplotlib.pyplot as plt


def detect_angle(image_path):
    """检测图片角度"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
        
        angles = []
        if lines is not None:
            for rho, theta in lines[:10]:
                angle = theta * 180 / np.pi
                # 将角度转换到-45到45度范围
                if angle > 45:
                    angle = angle - 90
                elif angle < -45:
                    angle = angle + 90
                angles.append(angle)
        
        if angles:
            correction_angle = np.median(angles)
            # 如果角度很小，不需要校正
            if abs(correction_angle) < 1:
                correction_angle = 0
            return correction_angle
        else:
            return 0
            
    except Exception as e:
        print(f"检测角度时出错 {image_path}: {e}")
        return 0


def correct_image_angle(img, angle):
    """校正图片角度"""
    if abs(angle) < 1:
        return img
    
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算新的边界框大小
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_width = int((height * sin_val) + (width * cos_val))
    new_height = int((height * cos_val) + (width * sin_val))
    
    # 调整旋转矩阵的平移部分
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # 执行旋转
    corrected_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height), 
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=(255, 255, 255))
    return corrected_img


def show_correction_comparison(original_img, corrected_img, angle, image_name):
    """显示校正前后对比"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'角度校正对比 - {image_name}', fontsize=14)
        
        # 原始图片
        ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'原始图片')
        ax1.axis('off')
        
        # 校正后的图片
        ax2.imshow(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'校正后 (旋转 {angle:.1f}°)')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"显示对比图时出错: {e}")


def calculate_histogram_similarity(img1, img2):
    """计算直方图相似度"""
    try:
        # 转换为HSV颜色空间
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        # 计算直方图
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        
        # 归一化直方图
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # 计算相关性
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, correlation)
        
    except Exception as e:
        print(f"计算直方图相似度时出错: {e}")
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


def main():
    """主函数"""
    # 配置参数
    target_image = 'sample1.jpg'
    search_folder = 'opt_result/21de6e8046d90a12a59e5017a09b189'
    
    print("开始带角度校正的图像相似度分析...")
    print(f"目标图片: {target_image}")
    print(f"搜索文件夹: {search_folder}")
    
    # 检查文件是否存在
    if not os.path.exists(target_image):
        print(f"目标图片不存在: {target_image}")
        return
    
    if not os.path.exists(search_folder):
        print(f"搜索文件夹不存在: {search_folder}")
        return
    
    # 读取并校正目标图片
    print("\n=== 处理目标图片 ===")
    target_img = cv2.imread(target_image)
    target_angle = detect_angle(target_image)
    target_corrected = correct_image_angle(target_img, target_angle)
    
    print(f"目标图片检测角度: {target_angle:.1f}°")
    
    if abs(target_angle) > 1:
        print("显示目标图片校正过程...")
        show_correction_comparison(target_img, target_corrected, target_angle, "目标图片")
    
    # 创建校正图片保存目录
    corrected_dir = "corrected_images"
    if not os.path.exists(corrected_dir):
        os.makedirs(corrected_dir)
    
    # 保存校正后的目标图片
    if abs(target_angle) > 1:
        target_corrected_path = os.path.join(corrected_dir, "sample1_corrected.jpg")
        cv2.imwrite(target_corrected_path, target_corrected)
        print(f"校正后的目标图片已保存: {target_corrected_path}")
    
    # 获取所有图片文件
    image_files = get_all_images(search_folder)
    if not image_files:
        print("未找到任何图片文件")
        return
    
    print(f"\n找到 {len(image_files)} 张图片，开始处理...")
    
    similarities = []
    
    for i, img_path in enumerate(image_files, 1):
        # 跳过目标图片本身
        if os.path.abspath(img_path) == os.path.abspath(target_image):
            continue
        
        print(f"\n处理 {i}/{len(image_files)}: {os.path.basename(img_path)}")
        
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print("  跳过: 无法读取图片")
            continue
        
        # 检测并校正角度
        angle = detect_angle(img_path)
        corrected_img = correct_image_angle(img, angle)
        
        print(f"  检测角度: {angle:.1f}°")
        
        # 保存校正后的图片
        if abs(angle) > 1:
            base_name = os.path.basename(img_path)
            name, ext = os.path.splitext(base_name)
            corrected_path = os.path.join(corrected_dir, f"{name}_corrected{ext}")
            cv2.imwrite(corrected_path, corrected_img)
            print(f"  校正后图片已保存: {corrected_path}")
        
        # 计算相似度
        similarity = calculate_histogram_similarity(target_corrected, corrected_img)
        similarity_percent = similarity * 100
        
        print(f"  相似度: {similarity_percent:.1f}%")
        
        similarities.append((img_path, similarity, similarity_percent, angle))
    
    if not similarities:
        print("没有成功处理的图片")
        return
    
    # 按相似度排序（从高到低）
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 显示结果
    print(f"\n{'='*80}")
    print("相似度排行榜 (前10名) - 直方图方法 + 角度校正")
    print(f"{'='*80}")
    print(f"{'排名':<4} {'相似度':<10} {'校正角度':<10} {'图片路径'}")
    print("-" * 80)
    
    top_10 = similarities[:10]
    for i, (img_path, similarity, similarity_percent, angle) in enumerate(top_10, 1):
        rel_path = os.path.relpath(img_path)
        print(f"{i:<4} {similarity_percent:.1f}%{'':<6} {angle:.1f}°{'':<6} {rel_path}")
    
    # 显示最相似的图片
    best_match = similarities[0]
    print(f"\n🎯 最相似的图片:")
    print(f"  📁 路径: {os.path.relpath(best_match[0])}")
    print(f"  💯 相似度: {best_match[2]:.1f}%")
    print(f"  🔄 校正角度: {best_match[3]:.1f}°")
    
    # 显示最相似图片的校正过程
    if abs(best_match[3]) > 1:
        print(f"\n显示最相似图片的校正过程...")
        best_img = cv2.imread(best_match[0])
        best_corrected = correct_image_angle(best_img, best_match[3])
        show_correction_comparison(best_img, best_corrected, best_match[3], 
                                 os.path.basename(best_match[0]))
    
    print(f"\n✅ 分析完成！")
    print(f"📊 所有校正后的图片已保存到 '{corrected_dir}' 文件夹")


if __name__ == '__main__':
    main() 