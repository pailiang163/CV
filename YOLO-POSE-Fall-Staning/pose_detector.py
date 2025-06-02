import cv2
import numpy as np
from ultralytics import YOLO
import math
from PIL import Image, ImageDraw, ImageFont
import os

class PoseDetector:
    def __init__(self, model_path='yolov12l-pose.pt'):
        """
        初始化姿态检测器
        Args:
            model_path: YOLO模型路径
        """
        self.model = YOLO(model_path)
        
        # COCO姿态关键点索引
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # 关键点连接关系（用于绘制骨架）
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        # 尝试加载中文字体
        self.font = None
        self.use_chinese = False
        try:
            # 尝试加载系统中文字体
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/simsun.ttc",  # 宋体
                "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
                "/System/Library/Fonts/PingFang.ttc",  # macOS
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    self.font = ImageFont.truetype(font_path, 24)
                    self.use_chinese = True
                    break
        except:
            pass
        
    def detect_poses(self, image):
        """
        检测图像中的人体姿态
        Args:
            image: 输入图像
        Returns:
            检测结果
        """
        results = self.model(image)
        return results
    
    def calculate_angle(self, p1, p2, p3):
        """
        计算三点之间的角度
        Args:
            p1, p2, p3: 三个点的坐标 (x, y)
        Returns:
            角度值（度）
        """
        if any(p is None for p in [p1, p2, p3]):
            return None
            
        # 计算向量
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # 计算角度
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def get_keypoint(self, keypoints, index):
        """
        获取指定索引的关键点坐标
        Args:
            keypoints: 关键点数组
            index: 关键点索引
        Returns:
            关键点坐标 (x, y) 或 None
        """
        if index < len(keypoints) and keypoints[index][2] > 0.5:  # 置信度阈值
            return (int(keypoints[index][0]), int(keypoints[index][1]))
        return None
    
    def is_keypoint_valid(self, keypoints, index):
        """
        检查关键点是否有效（坐标和不为0且置信度>0.5）
        Args:
            keypoints: 关键点数组
            index: 关键点索引
        Returns:
            bool: 关键点是否有效
        """
        if index >= len(keypoints):
            return False
        
        x, y, conf = keypoints[index]
        # 检查坐标和是否为0，以及置信度是否足够
        return (x + y) != 0 and conf > 0.5
    
    def analyze_pose(self, keypoints):
        """
        分析姿态，判断是站立、摔倒、坐着还是其他状态
        Args:
            keypoints: 关键点数组
        Returns:
            姿态状态和分析结果
        """
        # 初始化完整的分析结果结构
        analysis = {
            'status': 'other',
            'confidence': 0.0,
            'details': {},
            'keypoints_info': {},
            'center_points': {
                'shoulders_c': None,
                'hips_c': None,
                'knee_c': None,
                'ankle_c': None
            },
            'angles': {
                'body_angle': 0.0,
                'aspect_ratio': 0.0,
                'shoulder_hip_vertical_diff': 0.0,
                'hip_knee_shoulder_angle': None,
                'ankle_knee_hip_angle': None
            },
            'scores': {'Stand': 0, 'Fall': 0, 'Sit': 0, 'other': 0}
        }
        
        # 1. 关键点有效性检测
        key_indices = [5, 6, 11, 12, 13, 14, 15, 16]  # 双肩/髋/膝/踝
        ATHERPOSE = 0
        
        for idx in key_indices:
            if not self.is_keypoint_valid(keypoints, idx):
                ATHERPOSE += 1
        
        analysis['details']['invalid_keypoints'] = ATHERPOSE
        
        # 如果无效关键点≥8，直接标记为"other"
        if ATHERPOSE >= 8:
            analysis['status'] = 'other'
            analysis['confidence'] = 0.9
            analysis['details']['reason'] = '关键点缺失过多'
            return analysis
        
        # 2. 获取关键点坐标
        left_shoulder = self.get_keypoint(keypoints, 5)
        right_shoulder = self.get_keypoint(keypoints, 6)
        left_hip = self.get_keypoint(keypoints, 11)
        right_hip = self.get_keypoint(keypoints, 12)
        left_knee = self.get_keypoint(keypoints, 13)
        right_knee = self.get_keypoint(keypoints, 14)
        left_ankle = self.get_keypoint(keypoints, 15)
        right_ankle = self.get_keypoint(keypoints, 16)
        
        # 记录关键点信息
        analysis['keypoints_info'] = {
            'left_shoulder': left_shoulder,
            'right_shoulder': right_shoulder,
            'left_hip': left_hip,
            'right_hip': right_hip,
            'left_knee': left_knee,
            'right_knee': right_knee,
            'left_ankle': left_ankle,
            'right_ankle': right_ankle
        }
        
        # 3. 身体中心点计算
        shoulders_c = None
        hips_c = None
        knee_c = None
        ankle_c = None
        
        # 肩部中心
        if left_shoulder and right_shoulder:
            shoulders_c = ((left_shoulder[0] + right_shoulder[0]) // 2,
                          (left_shoulder[1] + right_shoulder[1]) // 2)
        elif left_shoulder:
            shoulders_c = left_shoulder
        elif right_shoulder:
            shoulders_c = right_shoulder
            
        # 髋部中心
        if left_hip and right_hip:
            hips_c = ((left_hip[0] + right_hip[0]) // 2,
                     (left_hip[1] + right_hip[1]) // 2)
        elif left_hip:
            hips_c = left_hip
        elif right_hip:
            hips_c = right_hip
            
        # 膝盖中心
        if left_knee and right_knee:
            knee_c = ((left_knee[0] + right_knee[0]) // 2,
                     (left_knee[1] + right_knee[1]) // 2)
        elif left_knee:
            knee_c = left_knee
        elif right_knee:
            knee_c = right_knee
            
        # 踝部中心
        if left_ankle and right_ankle:
            ankle_c = ((left_ankle[0] + right_ankle[0]) // 2,
                      (left_ankle[1] + right_ankle[1]) // 2)
        elif left_ankle:
            ankle_c = left_ankle
        elif right_ankle:
            ankle_c = right_ankle
        
        # 更新中心点信息
        analysis['center_points'].update({
            'shoulders_c': shoulders_c,
            'hips_c': hips_c,
            'knee_c': knee_c,
            'ankle_c': ankle_c
        })
        
        # 如果关键中心点缺失，返回other状态
        if not shoulders_c or not hips_c:
            analysis['status'] = 'other'
            analysis['confidence'] = 0.8
            analysis['details']['reason'] = '关键中心点缺失'
            return analysis
        
        # 4. 核心特征计算
        # a. 身体中心线角度：肩髋中心连线与水平线的夹角
        body_angle = math.degrees(math.atan2(
            abs(shoulders_c[1] - hips_c[1]),
            abs(shoulders_c[0] - hips_c[0])
        ))
        
        # b. 宽高比：检测框宽度/高度的比例
        # 计算人体边界框
        valid_points = [p for p in [left_shoulder, right_shoulder, left_hip, right_hip, 
                                   left_knee, right_knee, left_ankle, right_ankle] if p is not None]
        if valid_points:
            x_coords = [p[0] for p in valid_points]
            y_coords = [p[1] for p in valid_points]
            bbox_width = max(x_coords) - min(x_coords)
            bbox_height = max(y_coords) - min(y_coords)
            aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
        else:
            aspect_ratio = 0
        
        # c. 肩髋垂直距离差
        shoulder_hip_vertical_diff = abs(shoulders_c[1] - hips_c[1])
        
        # d. 关节角度
        hip_knee_shoulder_angle = None
        ankle_knee_hip_angle = None
        
        if knee_c and shoulders_c and hips_c:
            hip_knee_shoulder_angle = self.calculate_angle(hips_c, knee_c, shoulders_c)
        
        if ankle_c and knee_c and hips_c:
            ankle_knee_hip_angle = self.calculate_angle(ankle_c, knee_c, hips_c)
        
        # 更新角度信息
        analysis['angles'].update({
            'body_angle': body_angle,
            'aspect_ratio': aspect_ratio,
            'shoulder_hip_vertical_diff': shoulder_hip_vertical_diff,
            'hip_knee_shoulder_angle': hip_knee_shoulder_angle,
            'ankle_knee_hip_angle': ankle_knee_hip_angle
        })
        
        # 5. 状态评分体系
        status_score = {'Stand': 0, 'Fall': 0, 'Sit': 0, 'other': 0}
        
        # 评分条件1：身体角度判断
        if abs(body_angle) <= 25:  # 身体接近水平
            status_score['Fall'] += 0.8
            analysis['details']['body_angle_score'] = 'Fall +0.8 (接近水平)'
        elif abs(body_angle) >= 65:  # 身体接近垂直
            status_score['Stand'] += 0.8
            analysis['details']['body_angle_score'] = 'Stand +0.8 (接近垂直)'
        
        # 评分条件2：宽高比判断
        if aspect_ratio > 1.67:  # 宽度明显大于高度
            status_score['Fall'] += 0.8
            analysis['details']['aspect_ratio_score'] = 'Fall +0.8 (宽高比大)'
        elif aspect_ratio < 0.6:  # 高度明显大于宽度
            status_score['Stand'] += 0.6
            analysis['details']['aspect_ratio_score'] = 'Stand +0.6 (宽高比小)'
        
        # 评分条件3：踝膝髋角度判断
        if ankle_knee_hip_angle and 125 <= ankle_knee_hip_angle <= 180:
            status_score['Stand'] += 1.6
            analysis['details']['ankle_knee_hip_score'] = 'Stand +1.6 (腿部伸直)'
        elif ankle_knee_hip_angle and 60 <= ankle_knee_hip_angle <= 120:
            status_score['Sit'] += 1.2
            analysis['details']['ankle_knee_hip_score'] = 'Sit +1.2 (腿部弯曲)'
        
        # 评分条件4：膝部垂直位置判断
        if knee_c and shoulders_c:
            if knee_c[1] > shoulders_c[1]:  # 膝部低于肩部（y坐标更大）
                status_score['Fall'] += 0.6
                analysis['details']['knee_position_score'] = 'Fall +0.6 (膝部低于肩部)'
            else:
                status_score['Stand'] += 0.4
                analysis['details']['knee_position_score'] = 'Stand +0.4 (膝部高于肩部)'
        
        # 评分条件5：髋膝肩角度判断
        if hip_knee_shoulder_angle:
            if 150 <= hip_knee_shoulder_angle <= 180:
                status_score['Stand'] += 1.0
                analysis['details']['hip_knee_shoulder_score'] = 'Stand +1.0 (身体挺直)'
            elif 90 <= hip_knee_shoulder_angle <= 120:
                status_score['Sit'] += 1.0
                analysis['details']['hip_knee_shoulder_score'] = 'Sit +1.0 (身体弯曲)'
        
        # 评分条件6：肩髋垂直距离判断
        if shoulder_hip_vertical_diff < 30:  # 肩髋距离很小，可能是躺着
            status_score['Fall'] += 0.5
            analysis['details']['vertical_diff_score'] = 'Fall +0.5 (肩髋距离小)'
        elif shoulder_hip_vertical_diff > 80:  # 肩髋距离大，可能是站立
            status_score['Stand'] += 0.5
            analysis['details']['vertical_diff_score'] = 'Stand +0.5 (肩髋距离大)'
        
        # 评分条件7：关键点完整性奖励
        if ATHERPOSE <= 2:  # 关键点基本完整
            max_status = max(status_score, key=status_score.get)
            status_score[max_status] += 0.3
            analysis['details']['completeness_bonus'] = f'{max_status} +0.3 (关键点完整)'
        
        # 评分条件8：踝部位置判断
        if ankle_c and hips_c:
            if ankle_c[1] < hips_c[1]:  # 踝部高于髋部，可能是坐着或特殊姿势
                status_score['Sit'] += 0.4
                analysis['details']['ankle_position_score'] = 'Sit +0.4 (踝部高于髋部)'
        
        # 更新评分
        analysis['scores'] = status_score
        
        # 6. 最终判定
        max_score = max(status_score.values())
        if max_score == 0:
            analysis['status'] = 'other'
            analysis['confidence'] = 0.3
        else:
            final_status = max(status_score, key=status_score.get)
            analysis['status'] = final_status
            # 置信度基于最高分与次高分的差距
            sorted_scores = sorted(status_score.values(), reverse=True)
            if len(sorted_scores) > 1:
                confidence = min(0.95, max_score / (max_score + sorted_scores[1] + 0.1))
            else:
                confidence = min(0.95, max_score / (max_score + 0.1))
            analysis['confidence'] = confidence
        
        return analysis
    
    def draw_pose(self, image, keypoints, pose_analysis):
        """
        在图像上绘制姿态关键点和分析结果
        Args:
            image: 输入图像
            keypoints: 关键点数组
            pose_analysis: 姿态分析结果
        Returns:
            绘制后的图像
        """
        img = image.copy()
        
        # 绘制关键点
        for i, kpt in enumerate(keypoints):
            if kpt[2] > 0.5:  # 置信度阈值
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(img, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # 绘制骨架
        for connection in self.skeleton:
            kpt1_idx, kpt2_idx = connection[0] - 1, connection[1] - 1
            if (kpt1_idx < len(keypoints) and kpt2_idx < len(keypoints) and
                keypoints[kpt1_idx][2] > 0.5 and keypoints[kpt2_idx][2] > 0.5):
                
                x1, y1 = int(keypoints[kpt1_idx][0]), int(keypoints[kpt1_idx][1])
                x2, y2 = int(keypoints[kpt2_idx][0]), int(keypoints[kpt2_idx][1])
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 绘制中心点
        center_points = pose_analysis.get('center_points', {})
        for name, point in center_points.items():
            if point:
                cv2.circle(img, point, 8, (255, 255, 0), -1)
                cv2.putText(img, name[:3], (point[0]+10, point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # 显示分析结果
        status = pose_analysis.get('status', 'unknown')
        confidence = pose_analysis.get('confidence', 0.0)
        
        # 根据状态选择颜色和显示文本
        if self.use_chinese and self.font:
            # 使用中文显示
            status_map = {
                'Stand': ('站立', (0, 255, 0)),      # 绿色
                'Fall': ('摔倒', (0, 0, 255)),       # 红色
                'Sit': ('坐着', (255, 0, 255)),      # 紫色
                'other': ('其他', (0, 255, 255))     # 黄色
            }
        else:
            # 使用英文显示
            status_map = {
                'Stand': ('Standing', (0, 255, 0)),   # 绿色
                'Fall': ('Fallen', (0, 0, 255)),      # 红色
                'Sit': ('Sitting', (255, 0, 255)),    # 紫色
                'other': ('Other', (0, 255, 255))     # 黄色
            }
        
        display_text, color = status_map.get(status, ('Unknown', (128, 128, 128)))
        text = f"{display_text} ({confidence:.2f})"
        
        # 绘制状态文本
        img = self.put_chinese_text(img, text, (10, 30), color, 28)
        
        # 显示详细信息
        y_offset = 70
        
        # 安全地显示角度信息
        angles = pose_analysis.get('angles', {})
        if 'body_angle' in angles:
            cv2.putText(img, f"Body Angle: {angles['body_angle']:.1f}°", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        if 'aspect_ratio' in angles:
            cv2.putText(img, f"Aspect Ratio: {angles['aspect_ratio']:.2f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # 安全地显示评分
        scores = pose_analysis.get('scores', {})
        for status_name, score in scores.items():
            if score > 0:
                cv2.putText(img, f"{status_name}: {score:.1f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
        
        # 安全地显示无效关键点数量
        details = pose_analysis.get('details', {})
        invalid_count = details.get('invalid_keypoints', 0)
        cv2.putText(img, f"Invalid KPs: {invalid_count}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img
    
    def process_image(self, image_path):
        """
        处理单张图像
        Args:
            image_path: 图像路径
        Returns:
            处理结果
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        results = self.detect_poses(image)
        
        processed_results = []
        for result in results:
            if result.keypoints is not None:
                for keypoints in result.keypoints.data:
                    pose_analysis = self.analyze_pose(keypoints.cpu().numpy())
                    annotated_image = self.draw_pose(image, keypoints.cpu().numpy(), pose_analysis)
                    processed_results.append({
                        'image': annotated_image,
                        'analysis': pose_analysis,
                        'keypoints': keypoints.cpu().numpy()
                    })
        
        return processed_results
    
    def process_video(self, video_path, output_path=None):
        """
        处理视频文件
        Args:
            video_path: 视频路径
            output_path: 输出视频路径
        """
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.detect_poses(frame)
            
            for result in results:
                if result.keypoints is not None:
                    for keypoints in result.keypoints.data:
                        pose_analysis = self.analyze_pose(keypoints.cpu().numpy())
                        frame = self.draw_pose(frame, keypoints.cpu().numpy(), pose_analysis)
            
            if output_path:
                out.write(frame)
            
            cv2.imshow('姿态检测', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    
    def put_chinese_text(self, img, text, position, color=(255, 255, 255), font_size=24):
        """
        在图像上绘制中文文本
        Args:
            img: OpenCV图像
            text: 要显示的文本
            position: 文本位置 (x, y)
            color: 文本颜色 (B, G, R)
            font_size: 字体大小
        Returns:
            处理后的图像
        """
        if not self.use_chinese or self.font is None:
            # 如果没有中文字体，使用英文
            cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            return img
        
        # 转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 调整字体大小
        if font_size != 24:
            try:
                font = ImageFont.truetype(self.font.path, font_size)
            except:
                font = self.font
        else:
            font = self.font
        
        # 绘制文本 (PIL使用RGB颜色)
        rgb_color = (color[2], color[1], color[0])  # BGR转RGB
        draw.text(position, text, font=font, fill=rgb_color)
        
        # 转换回OpenCV图像
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv 