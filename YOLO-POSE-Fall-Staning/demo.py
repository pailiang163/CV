#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from pose_detector import PoseDetector

def create_demo_image():
    """
    创建一个演示图像（如果没有测试图像的话）
    """
    # 创建一个简单的演示图像
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # 添加文字说明
    cv2.putText(img, "请将您的测试图像放在项目目录中", (50, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "然后运行: python main.py -i your_image.jpg", (50, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "或使用摄像头: python main.py -c", (50, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return img

def demo_basic_usage():
    """
    基本使用演示
    """
    print("=== YOLO人体姿态检测演示 ===")
    print("正在初始化检测器...")
    
    try:
        # 初始化检测器
        detector = PoseDetector()
        print("✓ 检测器初始化成功")
        
        # 创建演示图像
        demo_img = create_demo_image()
        
        print("\n使用方法:")
        print("1. 处理图像文件:")
        print("   python main.py -i path/to/your/image.jpg")
        print("\n2. 处理视频文件:")
        print("   python main.py -i path/to/your/video.mp4 -o output.mp4")
        print("\n3. 实时摄像头检测:")
        print("   python main.py -c")
        print("\n4. 使用自定义模型:")
        print("   python main.py -i image.jpg -m yolov8s-pose.pt")
        
        print("\n姿态分析算法说明:")
        print("- 身体角度: 计算肩膀到臀部的倾斜角度")
        print("- 头部位置: 检查头部是否在肩膀上方")
        print("- 腿部角度: 分析腿部的弯曲程度")
        print("- 身体比例: 检查身体的高宽比")
        print("- 综合评分: 基于多个特征的综合判断")
        
        print("\n检测结果:")
        print("- 站立 (standing): 人体处于直立状态")
        print("- 摔倒 (fallen): 人体处于倒地状态")
        print("- 不确定 (uncertain): 无法明确判断状态")
        
        # 显示演示图像
        cv2.imshow("YOLO姿态检测演示", demo_img)
        print("\n按任意键关闭演示窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        print("请确保已安装所需依赖:")
        print("pip install -r requirements.txt")

def demo_keypoints_info():
    """
    显示关键点信息
    """
    print("\n=== COCO姿态关键点说明 ===")
    keypoint_names = [
        '0: nose (鼻子)', '1: left_eye (左眼)', '2: right_eye (右眼)', 
        '3: left_ear (左耳)', '4: right_ear (右耳)', '5: left_shoulder (左肩)',
        '6: right_shoulder (右肩)', '7: left_elbow (左肘)', '8: right_elbow (右肘)',
        '9: left_wrist (左腕)', '10: right_wrist (右腕)', '11: left_hip (左臀)',
        '12: right_hip (右臀)', '13: left_knee (左膝)', '14: right_knee (右膝)',
        '15: left_ankle (左踝)', '16: right_ankle (右踝)'
    ]
    
    for name in keypoint_names:
        print(f"  {name}")
    
    print("\n关键连接:")
    print("  头部: 鼻子 -> 眼睛 -> 耳朵")
    print("  躯干: 肩膀 -> 臀部")
    print("  手臂: 肩膀 -> 肘部 -> 手腕")
    print("  腿部: 臀部 -> 膝盖 -> 脚踝")

if __name__ == "__main__":
    demo_basic_usage()
    demo_keypoints_info() 