#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量图像分割脚本
使用SAM2对images文件夹中的图片进行分割，并将结果保存到以图像名称命名的文件夹中
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from sam2.sa.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    logger.error("SAM2未安装，请运行: pip install -r requirements.txt")
    exit(1)

class BatchImageSegmentor:
    def __init__(self, model_cfg="sam2_hiera_l.yaml", checkpoint_path=None):
        """
        初始化批量图像分割器
        
        Args:
            model_cfg: SAM2模型配置文件
            checkpoint_path: 模型权重文件路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 如果没有指定权重文件，尝试自动下载
        if checkpoint_path is None:
            checkpoint_path = self._download_checkpoint()
        
        try:
            # 构建SAM2模型
            sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            logger.info("SAM2模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _download_checkpoint(self):
        """下载SAM2权重文件"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / "sam2_hiera_large.pt"
        
        if not checkpoint_path.exists():
            logger.info("正在下载SAM2权重文件...")
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
            try:
                urllib.request.urlretrieve(url, checkpoint_path)
                logger.info("权重文件下载完成")
            except Exception as e:
                logger.error(f"权重文件下载失败: {e}")
                logger.info("请手动下载权重文件到checkpoints文件夹")
                raise
        
        return str(checkpoint_path)
    
    def segment_image(self, image_path, output_dir):
        """
        分割单张图像
        
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
        """
        try:
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                return False
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 设置图像到预测器
            self.predictor.set_image(image_rgb)
            
            # 生成自动分割点（使用网格采样）
            height, width = image_rgb.shape[:2]
            
            # 在图像上创建网格点
            grid_size = 32
            points = []
            labels = []
            
            for y in range(grid_size, height - grid_size, grid_size):
                for x in range(grid_size, width - grid_size, grid_size):
                    points.append([x, y])
                    labels.append(1)  # 前景点
            
            if not points:
                logger.warning(f"图像 {image_path} 太小，无法生成分割点")
                return False
            
            points = np.array(points)
            labels = np.array(labels)
            
            # 进行预测
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )
            
            # 保存分割结果
            self._save_segmentation_results(image_rgb, masks, scores, output_dir, image_path.stem)
            
            logger.info(f"图像 {image_path.name} 分割完成，共生成 {len(masks)} 个掩码")
            return True
            
        except Exception as e:
            logger.error(f"分割图像 {image_path} 时出错: {e}")
            return False
    
    def _save_segmentation_results(self, image, masks, scores, output_dir, image_name):
        """
        保存分割结果
        
        Args:
            image: 原始图像
            masks: 分割掩码
            scores: 分割得分
            output_dir: 输出目录
            image_name: 图像名称
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原始图像
        original_path = output_dir / f"{image_name}_original.jpg"
        Image.fromarray(image).save(original_path)
        
        # 为每个掩码创建可视化图像
        for i, (mask, score) in enumerate(zip(masks, scores)):
            # 创建掩码可视化
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原始图像
            axes[0].imshow(image)
            axes[0].set_title("原始图像")
            axes[0].axis('off')
            
            # 掩码
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title(f"掩码 {i+1} (得分: {score:.3f})")
            axes[1].axis('off')
            
            # 掩码覆盖在原图上
            overlay = image.copy()
            overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title("掩码覆盖")
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # 保存图像
            result_path = output_dir / f"{image_name}_mask_{i+1}_score_{score:.3f}.png"
            plt.savefig(result_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # 保存单独的掩码图像
            mask_path = output_dir / f"{image_name}_mask_{i+1}.png"
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    
    def process_images_folder(self, images_folder="../images", output_base_dir="../batch_output"):
        """
        批量处理images文件夹中的图像
        
        Args:
            images_folder: 输入图像文件夹路径
            output_base_dir: 输出基础目录
        """
        images_folder = Path(images_folder)
        output_base_dir = Path(output_base_dir)
        
        if not images_folder.exists():
            logger.error(f"图像文件夹不存在: {images_folder}")
            return
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 获取所有图像文件
        image_files = [
            f for f in images_folder.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            logger.warning(f"在文件夹 {images_folder} 中未找到图像文件")
            return
        
        logger.info(f"找到 {len(image_files)} 张图像，开始批量分割...")
        
        successful_count = 0
        failed_count = 0
        
        for image_file in image_files:
            logger.info(f"正在处理: {image_file.name}")
            
            # 为每张图像创建单独的输出文件夹
            image_output_dir = output_base_dir / image_file.stem
            
            if self.segment_image(image_file, image_output_dir):
                successful_count += 1
            else:
                failed_count += 1
        
        logger.info(f"批量分割完成！成功: {successful_count}，失败: {failed_count}")

def main():
    """主函数"""
    logger.info("开始批量图像分割...")
    
    try:
        # 创建分割器实例
        segmentor = BatchImageSegmentor()
        
        # 处理images文件夹中的图像
        segmentor.process_images_folder()
        
        logger.info("所有图像处理完成！")
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise

if __name__ == "__main__":
    main() 