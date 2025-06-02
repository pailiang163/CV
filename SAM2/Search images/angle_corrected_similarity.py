import os
import glob
import cv2
import numpy as np
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math


def detect_and_correct_angle(image_path, display=True):
    """
    检测并校正图片角度
    :param image_path: 图片路径
    :param display: 是否显示校正过程
    :return: 校正后的图片和角度信息
    """
    try:
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            return None, 0
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        angles = []
        if lines is not None:
            for rho, theta in lines[:10]:  # 只取前10条线
                angle = theta * 180 / np.pi
                # 将角度转换到-45到45度范围
                if angle > 45:
                    angle = angle - 90
                elif angle < -45:
                    angle = angle + 90
                angles.append(angle)
        
        # 计算平均角度
        if angles:
            # 使用中位数来减少异常值影响
            correction_angle = np.median(angles)
        else:
            correction_angle = 0
        
        # 如果角度很小，不需要校正
        if abs(correction_angle) < 1:
            correction_angle = 0
        
        # 旋转图片
        if correction_angle != 0:
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            
            # 计算旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
            
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
        else:
            corrected_img = img.copy()
        
        # 显示校正过程
        if display and correction_angle != 0:
            display_correction_process(img, corrected_img, correction_angle, image_path, edges, lines)
        
        return corrected_img, correction_angle
        
    except Exception as e:
        print(f"角度校正时出错 {image_path}: {e}")
        return None, 0


def display_correction_process(original_img, corrected_img, angle, image_path, edges, lines):
    """显示角度校正过程"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'图片角度校正过程 - {os.path.basename(image_path)}', fontsize=16)
        
        # 原始图片
        axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f'原始图片')
        axes[0, 0].axis('off')
        
        # 边缘检测结果
        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title('边缘检测结果')
        axes[0, 1].axis('off')
        
        # 检测到的直线
        line_img = cv2.cvtColor(original_img.copy(), cv2.COLOR_BGR2RGB)
        if lines is not None:
            for rho, theta in lines[:5]:  # 显示前5条线
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        axes[1, 0].imshow(line_img)
        axes[1, 0].set_title(f'检测到的直线 (角度: {angle:.2f}°)')
        axes[1, 0].axis('off')
        
        # 校正后的图片
        axes[1, 1].imshow(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f'校正后图片 (旋转 {angle:.2f}°)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"显示校正过程时出错: {e}")


def save_corrected_image(corrected_img, original_path, output_dir="corrected_images"):
    """保存校正后的图片"""
    try:
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 生成输出文件名
        base_name = os.path.basename(original_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_corrected{ext}")
        
        # 保存图片
        cv2.imwrite(output_path, corrected_img)
        return output_path
        
    except Exception as e:
        print(f"保存校正图片时出错: {e}")
        return None


def calculate_image_hash_from_array(img_array):
    """从numpy数组计算图片哈希值"""
    try:
        # 将BGR转换为RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        # 转换为PIL Image
        pil_img = Image.fromarray(img_rgb)
        return imagehash.dhash(pil_img)
    except Exception as e:
        print(f"计算哈希值时出错: {e}")
        return None


def calculate_histogram_similarity_from_array(img1_array, img2_array):
    """从numpy数组计算直方图相似度"""
    try:
        if img1_array is None or img2_array is None:
            return 0
        
        # 转换为HSV颜色空间
        hsv1 = cv2.cvtColor(img1_array, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2_array, cv2.COLOR_BGR2HSV)
        
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


def calculate_template_matching_from_array(img1_array, img2_array):
    """从numpy数组计算模板匹配相似度"""
    try:
        if img1_array is None or img2_array is None:
            return 0
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(img1_array, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_array, cv2.COLOR_BGR2GRAY)
        
        # 如果模板比目标图片大，交换它们
        if gray2.shape[0] > gray1.shape[0] or gray2.shape[1] > gray1.shape[1]:
            gray1, gray2 = gray2, gray1
        
        # 执行模板匹配
        result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
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


def find_similar_images_with_correction(target_image, search_folder, method='histogram', 
                                       top_n=5, show_correction=True, save_corrected=False):
    """
    使用角度校正后进行图像相似度比较
    :param target_image: 目标图片路径
    :param search_folder: 搜索文件夹路径
    :param method: 相似度计算方法 ('hash', 'histogram', 'template')
    :param top_n: 返回前N个最相似的结果
    :param show_correction: 是否显示校正过程
    :param save_corrected: 是否保存校正后的图片
    """
    print(f"目标图片: {target_image}")
    print(f"搜索文件夹: {search_folder}")
    print(f"计算方法: {method}")
    print(f"角度校正: 启用")
    
    # 检查文件是否存在
    if not os.path.exists(target_image):
        print(f"目标图片不存在: {target_image}")
        return []
    
    if not os.path.exists(search_folder):
        print(f"搜索文件夹不存在: {search_folder}")
        return []
    
    # 校正目标图片
    print("\n=== 校正目标图片 ===")
    target_corrected, target_angle = detect_and_correct_angle(target_image, display=show_correction)
    if target_corrected is None:
        print("无法处理目标图片")
        return []
    
    print(f"目标图片角度校正: {target_angle:.2f}°")
    
    # 保存校正后的目标图片
    if save_corrected and target_angle != 0:
        target_corrected_path = save_corrected_image(target_corrected, target_image)
        if target_corrected_path:
            print(f"校正后的目标图片已保存: {target_corrected_path}")
    
    # 获取所有图片文件
    image_files = get_all_images(search_folder)
    if not image_files:
        print("未找到任何图片文件")
        return []
    
    print(f"\n找到 {len(image_files)} 张图片，开始角度校正和相似度计算...")
    
    similarities = []
    corrected_images = []
    
    # 如果使用哈希方法，先计算目标图片的哈希值
    target_hash = None
    if method == 'hash':
        target_hash = calculate_image_hash_from_array(target_corrected)
        if target_hash is None:
            print("无法计算目标图片哈希值")
            return []
    
    for i, img_path in enumerate(image_files, 1):
        # 跳过目标图片本身
        if os.path.abspath(img_path) == os.path.abspath(target_image):
            continue
        
        print(f"\n处理图片 {i}/{len(image_files)}: {os.path.basename(img_path)}")
        
        # 校正当前图片
        corrected_img, correction_angle = detect_and_correct_angle(img_path, display=False)
        if corrected_img is None:
            print(f"  跳过: 无法处理图片")
            continue
        
        print(f"  角度校正: {correction_angle:.2f}°")
        
        # 保存校正后的图片
        if save_corrected and correction_angle != 0:
            corrected_path = save_corrected_image(corrected_img, img_path)
            if corrected_path:
                print(f"  校正后图片已保存: {corrected_path}")
        
        # 计算相似度
        if method == 'hash':
            img_hash = calculate_image_hash_from_array(corrected_img)
            if img_hash is not None and target_hash is not None:
                distance = target_hash - img_hash  # 汉明距离
                similarity_score = max(0, (64 - distance) / 64 * 100)
                similarities.append((img_path, distance, similarity_score, correction_angle))
        
        elif method == 'histogram':
            hist_similarity = calculate_histogram_similarity_from_array(target_corrected, corrected_img)
            similarity_score = hist_similarity * 100
            distance = 1 - hist_similarity
            similarities.append((img_path, distance, similarity_score, correction_angle))
        
        elif method == 'template':
            template_similarity = calculate_template_matching_from_array(target_corrected, corrected_img)
            similarity_score = template_similarity * 100
            distance = 1 - template_similarity
            similarities.append((img_path, distance, similarity_score, correction_angle))
        
        print(f"  相似度: {similarity_score:.1f}%")
        
        # 显示进度
        if i % 10 == 0 or i == len(image_files):
            print(f"\n总进度: {i}/{len(image_files)} ({i/len(image_files)*100:.1f}%)")
    
    if not similarities:
        print("没有成功处理的图片")
        return []
    
    # 按距离排序（距离越小越相似）
    similarities.sort(key=lambda x: x[1])
    
    # 显示结果
    print(f"\n{'='*100}")
    print(f"相似度排行榜 (前{top_n}名) - 方法: {method.upper()} + 角度校正")
    print(f"{'='*100}")
    print(f"{'排名':<4} {'距离/差异':<12} {'相似度':<10} {'校正角度':<10} {'图片路径'}")
    print("-" * 100)
    
    top_results = similarities[:top_n]
    for i, (img_path, distance, similarity, angle) in enumerate(top_results, 1):
        rel_path = os.path.relpath(img_path)
        if method == 'hash':
            distance_str = f"{int(distance)}"
        else:
            distance_str = f"{distance:.4f}"
        print(f"{i:<4} {distance_str:<12} {similarity:.1f}%{'':<6} {angle:.1f}°{'':<6} {rel_path}")
    
    # 显示最相似的图片
    if top_results:
        best_match = top_results[0]
        print(f"\n🎯 最相似的图片:")
        print(f"  📁 路径: {os.path.relpath(best_match[0])}")
        if method == 'hash':
            print(f"  📏 汉明距离: {int(best_match[1])}")
        else:
            print(f"  📏 差异度: {best_match[1]:.4f}")
        print(f"  💯 相似度: {best_match[2]:.1f}%")
        print(f"  🔄 校正角度: {best_match[3]:.1f}°")
        
        # 显示最相似图片的校正过程
        if show_correction and abs(best_match[3]) > 1:
            print(f"\n显示最相似图片的校正过程...")
            detect_and_correct_angle(best_match[0], display=True)
    
    return top_results


if __name__ == '__main__':
    # 配置参数
    target_image = 'sample1.jpg'
    search_folder = 'opt_result/21de6e8046d90a12a59e5017a09b189'
    
    # 选择方法: 'hash', 'histogram', 'template'
    method = 'histogram'
    
    print("🚀 开始带角度校正的图像相似度分析...")
    print("=" * 80)
    
    results = find_similar_images_with_correction(
        target_image=target_image,
        search_folder=search_folder,
        method=method,
        top_n=10,
        show_correction=True,  # 显示校正过程
        save_corrected=True    # 保存校正后的图片
    )
    
    if results:
        print(f"\n✅ 分析完成！找到 {len(results)} 个相似的图片")
        print(f"📊 使用了角度校正技术提高比较准确性")
    else:
        print(f"\n❌ 分析失败或未找到相似图片") 