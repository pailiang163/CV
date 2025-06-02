import os
import glob
from PIL import Image
import imagehash
from typing import List, Tuple
import time


class ImageSimilarityFinder:
    """图像相似度查找器"""
    
    def __init__(self, hash_method='dhash'):
        """
        初始化相似度查找器
        :param hash_method: 哈希方法 ('dhash', 'phash', 'ahash', 'whash')
        """
        self.hash_methods = {
            'dhash': imagehash.dhash,
            'phash': imagehash.phash,
            'ahash': imagehash.average_hash,
            'whash': imagehash.whash
        }
        self.hash_method = self.hash_methods.get(hash_method, imagehash.dhash)
    
    def calculate_image_hash(self, img_path: str):
        """
        计算图片哈希值
        :param img_path: 图片路径
        :return: 图片哈希值
        """
        try:
            with Image.open(img_path) as img:
                return self.hash_method(img)
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")
            return None
    
    def hamming_distance(self, hash1, hash2) -> int:
        """
        计算汉明距离
        :param hash1: 第一个哈希值
        :param hash2: 第二个哈希值
        :return: 汉明距离（越小越相似）
        """
        if hash1 is None or hash2 is None:
            return float('inf')
        return hash1 - hash2
    
    def get_image_files(self, folder_path: str) -> List[str]:
        """
        获取文件夹中所有图片文件
        :param folder_path: 文件夹路径
        :return: 图片文件路径列表
        """
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
        image_files = []
        
        for ext in image_extensions:
            pattern = os.path.join(folder_path, '**', ext)
            image_files.extend(glob.glob(pattern, recursive=True))
            # 也搜索当前目录
            pattern = os.path.join(folder_path, ext)
            image_files.extend(glob.glob(pattern))
        
        # 去重并排序
        return sorted(list(set(image_files)))
    
    def find_most_similar(self, target_image: str, search_folder: str, top_n: int = 5) -> List[Tuple[str, int, float]]:
        """
        找到最相似的图片
        :param target_image: 目标图片路径
        :param search_folder: 搜索文件夹路径
        :param top_n: 返回前N个最相似的结果
        :return: [(图片路径, 汉明距离, 相似度百分比), ...]
        """
        print(f"正在计算目标图片 '{target_image}' 的哈希值...")
        target_hash = self.calculate_image_hash(target_image)
        
        if target_hash is None:
            print(f"无法处理目标图片: {target_image}")
            return []
        
        print(f"正在搜索文件夹 '{search_folder}' 中的图片...")
        image_files = self.get_image_files(search_folder)
        
        if not image_files:
            print(f"在文件夹 '{search_folder}' 中未找到图片文件")
            return []
        
        print(f"找到 {len(image_files)} 张图片，开始计算相似度...")
        
        similarities = []
        processed = 0
        
        for img_path in image_files:
            # 跳过目标图片本身
            if os.path.abspath(img_path) == os.path.abspath(target_image):
                continue
                
            img_hash = self.calculate_image_hash(img_path)
            if img_hash is not None:
                distance = self.hamming_distance(target_hash, img_hash)
                # 计算相似度百分比 (64位哈希，距离越小相似度越高)
                similarity_percent = max(0, (64 - distance) / 64 * 100)
                similarities.append((img_path, distance, similarity_percent))
            
            processed += 1
            if processed % 10 == 0:
                print(f"已处理 {processed}/{len(image_files)} 张图片...")
        
        # 按汉明距离排序（距离越小越相似）
        similarities.sort(key=lambda x: x[1])
        
        return similarities[:top_n]
    
    def print_results(self, results: List[Tuple[str, int, float]], target_image: str):
        """
        打印结果
        :param results: 相似度结果列表
        :param target_image: 目标图片路径
        """
        print(f"\n{'='*60}")
        print(f"目标图片: {target_image}")
        print(f"{'='*60}")
        
        if not results:
            print("未找到相似的图片")
            return
        
        print(f"找到 {len(results)} 个最相似的图片:")
        print(f"{'排名':<4} {'汉明距离':<8} {'相似度':<8} {'图片路径'}")
        print("-" * 80)
        
        for i, (img_path, distance, similarity) in enumerate(results, 1):
            # 获取相对路径以便显示
            rel_path = os.path.relpath(img_path)
            print(f"{i:<4} {distance:<8} {similarity:.1f}%{'':<4} {rel_path}")
        
        # 显示最相似的图片信息
        best_match = results[0]
        print(f"\n🎯 最相似的图片:")
        print(f"   路径: {os.path.relpath(best_match[0])}")
        print(f"   汉明距离: {best_match[1]}")
        print(f"   相似度: {best_match[2]:.1f}%")


def main():
    """主函数"""
    # 配置参数
    target_image = 'sample1.jpg'
    search_folder = 'opt_result'  # 可以修改为具体的img_path2文件夹路径
    
    # 如果要搜索特定的子文件夹，可以这样设置：
    # search_folder = 'opt_result/2ded08aa754066e9235454e451f5703'
    
    print("🔍 图像相似度查找器")
    print(f"目标图片: {target_image}")
    print(f"搜索文件夹: {search_folder}")
    
    # 检查目标图片是否存在
    if not os.path.exists(target_image):
        print(f"❌ 目标图片不存在: {target_image}")
        return
    
    # 检查搜索文件夹是否存在
    if not os.path.exists(search_folder):
        print(f"❌ 搜索文件夹不存在: {search_folder}")
        return
    
    # 创建相似度查找器
    finder = ImageSimilarityFinder(hash_method='dhash')  # 可以改为 'phash', 'ahash', 'whash'
    
    # 开始计时
    start_time = time.time()
    
    # 查找最相似的图片
    results = finder.find_most_similar(target_image, search_folder, top_n=10)
    
    # 结束计时
    end_time = time.time()
    
    # 打印结果
    finder.print_results(results, target_image)
    
    print(f"\n⏱️  总耗时: {end_time - start_time:.2f} 秒")


if __name__ == '__main__':
    main() 