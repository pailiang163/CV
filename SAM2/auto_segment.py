#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动图像分割脚本
使用SAM2的自动掩码生成器对images文件夹中的图片进行全自动分割
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
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    logger.error("SAM2未安装，请运行: pip install -r requirements.txt")
    exit(1)

class AutoSegmentor:
    def __init__(self, model_cfg="sam2_hiera_l.yaml", checkpoint_path=None):
        """
        初始化自动分割器
        
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
            
            # 创建自动掩码生成器
            self.mask_generator = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                points_per_side=32,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.95,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            logger.info("SAM2自动掩码生成器加载成功")
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
        自动分割单张图像
        
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
            
            # 生成自动掩码
            logger.info(f"正在为 {image_path.name} 生成掩码...")
            masks = self.mask_generator.generate(image_rgb)
            
            if not masks:
                logger.warning(f"图像 {image_path.name} 未生成任何掩码")
                return False
            
            # 按掩码面积排序（从大到小）
            masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            
            # 保存分割结果
            self._save_segmentation_results(image_rgb, masks, output_dir, image_path.stem)
            
            logger.info(f"图像 {image_path.name} 分割完成，共生成 {len(masks)} 个掩码")
            return True
            
        except Exception as e:
            logger.error(f"分割图像 {image_path} 时出错: {e}")
            return False
    
    def _save_segmentation_results(self, image, masks, output_dir, image_name):
        """
        保存分割结果
        
        Args:
            image: 原始图像
            masks: 自动生成的掩码列表
            output_dir: 输出目录
            image_name: 图像名称
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原始图像
        original_path = output_dir / f"{image_name}_original.jpg"
        Image.fromarray(image).save(original_path)
        
        # 创建全览图
        self._create_overview_image(image, masks, output_dir, image_name)
        
        # 保存每个掩码
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            area = mask_data['area']
            stability_score = mask_data['stability_score']
            
            # 保存单独的掩码图像
            mask_path = output_dir / f"{image_name}_mask_{i+1:03d}_area_{area}.png"
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
            
            # 创建掩码覆盖图像
            overlay = image.copy()
            overlay[mask] = overlay[mask] * 0.6 + np.array([255, 0, 0]) * 0.4
            overlay_path = output_dir / f"{image_name}_overlay_{i+1:03d}.jpg"
            Image.fromarray(overlay.astype(np.uint8)).save(overlay_path)
    
    def _create_overview_image(self, image, masks, output_dir, image_name):
        """
        创建包含所有掩码的概览图
        
        Args:
            image: 原始图像
            masks: 掩码列表
            output_dir: 输出目录
            image_name: 图像名称
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # 原始图像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("原始图像")
        axes[0, 0].axis('off')
        
        # 所有掩码叠加
        overlay = image.copy()
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(masks))))
        
        for i, mask_data in enumerate(masks[:20]):  # 只显示前20个掩码
            mask = mask_data['segmentation']
            color = colors[i % len(colors)][:3] * 255
            overlay[mask] = overlay[mask] * 0.7 + color * 0.3
        
        axes[0, 1].imshow(overlay.astype(np.uint8))
        axes[0, 1].set_title(f"所有掩码叠加 (共{len(masks)}个)")
        axes[0, 1].axis('off')
        
        # 掩码数量最大的几个
        if len(masks) > 0:
            largest_mask = masks[0]['segmentation']
            axes[1, 0].imshow(largest_mask, cmap='gray')
            axes[1, 0].set_title(f"最大掩码 (面积: {masks[0]['area']})")
            axes[1, 0].axis('off')
        
        # 掩码面积分布图
        if len(masks) > 1:
            areas = [mask['area'] for mask in masks]
            axes[1, 1].hist(areas, bins=20, alpha=0.7)
            axes[1, 1].set_title("掩码面积分布")
            axes[1, 1].set_xlabel("面积")
            axes[1, 1].set_ylabel("数量")
        
        plt.tight_layout()
        
        # 保存概览图
        overview_path = output_dir / f"{image_name}_overview.png"
        plt.savefig(overview_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_images_folder(self, images_folder="images", output_base_dir="output"):
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
        
        logger.info(f"找到 {len(image_files)} 张图像，开始自动分割...")
        
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
        
        logger.info(f"自动分割完成！成功: {successful_count}，失败: {failed_count}")

def main():
    """主函数"""
    logger.info("开始自动图像分割...")
    
    try:
        # 创建自动分割器实例
        segmentor = AutoSegmentor()
        
        # 处理images文件夹中的图像
        segmentor.process_images_folder()
        
        logger.info("所有图像处理完成！")
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise

if __name__ == "__main__":
    main() 