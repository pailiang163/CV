#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import ctypes
import argparse
import os
from pose_detector import PoseDetector

def print_detailed_analysis(analysis, person_id):
    """
    打印详细的姿态分析结果
    Args:
        analysis: 分析结果字典
        person_id: 人员编号
    """
    print(f"\n=== 检测到的人 {person_id} ===")
    print(f"最终状态: {analysis['status']}")
    print(f"置信度: {analysis['confidence']:.3f}")
    
    # 显示关键点信息
    print("\n关键点坐标:")
    keypoints_info = analysis['keypoints_info']
    for name, point in keypoints_info.items():
        if point:
            print(f"  {name}: ({point[0]}, {point[1]})")
        else:
            print(f"  {name}: 未检测到")
    
    # 显示中心点信息
    print("\n身体中心点:")
    center_points = analysis['center_points']
    for name, point in center_points.items():
        if point:
            print(f"  {name}: ({point[0]}, {point[1]})")
        else:
            print(f"  {name}: 未计算")
    
    # 显示角度和特征信息
    print("\n核心特征:")
    angles = analysis['angles']
    print(f"  身体角度: {angles['body_angle']:.1f}°")
    print(f"  宽高比: {angles['aspect_ratio']:.3f}")
    print(f"  肩髋垂直距离: {angles['shoulder_hip_vertical_diff']:.1f}")
    if angles['hip_knee_shoulder_angle']:
        print(f"  髋膝肩角度: {angles['hip_knee_shoulder_angle']:.1f}°")
    if angles['ankle_knee_hip_angle']:
        print(f"  踝膝髋角度: {angles['ankle_knee_hip_angle']:.1f}°")
    
    # 显示评分详情
    print("\n状态评分:")
    scores = analysis['scores']
    for status, score in scores.items():
        print(f"  {status}: {score:.2f}")
    
    # 显示评分详细过程
    print("\n评分详情:")
    details = analysis['details']
    invalid_count = details.get('invalid_keypoints', 0)
    print(f"  无效关键点数量: {invalid_count}")
    
    if 'reason' in details:
        print(f"  判定原因: {details['reason']}")
    
    # 显示各项评分
    score_items = [
        'body_angle_score', 'aspect_ratio_score', 'ankle_knee_hip_score',
        'knee_position_score', 'hip_knee_shoulder_score', 'vertical_diff_score',
        'completeness_bonus', 'ankle_position_score'
    ]
    
    for item in score_items:
        if item in details:
            print(f"  {item}: {details[item]}")

def main():
    parser = argparse.ArgumentParser(description='YOLO人体姿态检测 - 判断站立/摔倒/坐着')
    parser.add_argument('--input', '-i', default='images/people(26).jpg',help='输入文件路径（图像或视频）')
    parser.add_argument('--output', '-o', help='输出文件路径（仅对视频有效）')
    parser.add_argument('--model', '-m', default='yolo11l-pose.pt', help='YOLO模型路径')
    parser.add_argument('--camera', '-c', action='store_true', help='使用摄像头实时检测')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细分析信息')
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.camera and not args.input:
        parser.error("必须指定输入文件 (-i) 或使用摄像头 (-c)")
    
    # 初始化姿态检测器
    print("正在初始化YOLO姿态检测器...")
    detector = PoseDetector(args.model)
    print("初始化完成！")
    
    if args.camera:
        # 摄像头实时检测
        print("启动摄像头实时检测...")
        print("按 'q' 键退出，按 'v' 键切换详细信息显示")
        cap = cv2.VideoCapture(0)
        show_verbose = args.verbose
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            results = detector.detect_poses(frame)
            
            for result in results:
                if result.keypoints is not None:
                    for i, keypoints in enumerate(result.keypoints.data):
                        pose_analysis = detector.analyze_pose(keypoints.cpu().numpy())
                        frame = detector.draw_pose(frame, keypoints.cpu().numpy(), pose_analysis)
                        
                        # 打印检测结果
                        status = pose_analysis['status']
                        confidence = pose_analysis['confidence']
                        if status != 'other' or confidence > 0.5:
                            status_map = {
                                'Stand': '站立', 'Fall': '摔倒', 
                                'Sit': '坐着', 'other': '其他'
                            }
                            chinese_status = status_map.get(status, status)
                            print(f"检测结果: {chinese_status} (置信度: {confidence:.2f})")
                            
                            if show_verbose:
                                print_detailed_analysis(pose_analysis, i+1)
            
            cv2.imshow('实时姿态检测', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                show_verbose = not show_verbose
                print(f"详细信息显示: {'开启' if show_verbose else '关闭'}")
        
        cap.release()
        cv2.destroyAllWindows()
        
    elif os.path.isfile(args.input):
        # 检查文件类型
        file_ext = os.path.splitext(args.input)[1].lower()
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            # 处理图像
            print(f"正在处理图像: {args.input}")
            results = detector.process_image(args.input)
            
            if results:
                for i, result in enumerate(results):
                    analysis = result['analysis']
                    status_map = {
                        'Stand': '站立', 'Fall': '摔倒', 
                        'Sit': '坐着', 'other': '其他'
                    }
                    chinese_status = status_map.get(analysis['status'], analysis['status'])
                    
                    print(f"\n检测到的人 {i+1}:")
                    print(f"  状态: {chinese_status}")
                    print(f"  置信度: {analysis['confidence']:.3f}")
                    
                    if args.verbose:
                        print_detailed_analysis(analysis, i+1)
                    
                    # 显示结果、
                    
                    cv2.imshow(f'检测结果 {i+1}', result['image'])
                
                print("\n按任意键关闭窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("未检测到人体姿态")
                
        elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # 处理视频
            print(f"正在处理视频: {args.input}")
            print("按 'q' 键退出视频处理")
            detector.process_video(args.input, args.output)
            
        else:
            print(f"不支持的文件格式: {file_ext}")
    else:
        print(f"文件不存在: {args.input}")

if __name__ == "__main__":
    main() 