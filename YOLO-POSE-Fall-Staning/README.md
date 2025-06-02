# YOLO人体姿态检测 - 多状态姿态分析系统

基于YOLOv11/YOLOv12-pose的高精度人体姿态检测系统，能够智能判断人体的站立、摔倒、坐着等多种状态。

## 🎯 功能特点

- 🎯 **精确检测**: 使用最新的YOLO-pose模型进行人体关键点检测
- 🧠 **智能分析**: 基于多特征融合的高级姿态分析算法
- 📊 **多状态识别**: 支持站立、摔倒、坐着、其他四种状态判断
- 📸 **多种输入**: 支持图像、视频和实时摄像头
- 🎨 **可视化**: 实时显示关键点、骨架、中心点和分析结果
- ⚡ **高效处理**: 优化的算法确保实时性能
- 📈 **详细分析**: 提供完整的评分过程和特征数据

## 🔬 算法原理

### 1. 关键点有效性检测
- 检查8个核心关键点（双肩/髋/膝/踝）的有效性
- 坐标和为0表示关键点未检测到
- 无效关键点≥8时标记为"other"状态

### 2. 身体中心点计算
计算四个关键区域的中心坐标：
- **Shoulders_c**: (左肩+右肩)/2 - 肩部中心
- **Hips_c**: (左髋+右髋)/2 - 髋部中心  
- **Knee_c**: (左膝+右膝)/2 - 膝盖中心
- **Ankle_c**: (左踝+右踝)/2 - 踝部中心

### 3. 核心特征计算
- **身体中心线角度**: 肩髋中心连线与水平线的夹角
- **宽高比**: 检测框宽度/高度的比例
- **肩髋垂直距离差**: 肩部与髋部的y坐标差值
- **关节角度**: 
  - Hip-Knee-Shoulders角度(骨盆-膝-肩)
  - Ankle-Knee-Hip角度(踝-膝-骨盆)

### 4. 状态评分体系
初始化四个状态的基准分：
```python
status_score = {'Stand':0, 'Fall':0, 'Sit':0, 'other':0}
```

通过多个条件分支进行动态评分：

#### 主要评分条件：
- **身体角度判断**: ±25°内为摔倒倾向，≥65°为站立倾向
- **宽高比判断**: >1.67为摔倒倾向，<0.6为站立倾向  
- **踝膝髋角度**: 125-180°为站立，60-120°为坐着
- **膝部位置**: 膝部低于肩部为摔倒倾向
- **髋膝肩角度**: 150-180°为站立，90-120°为坐着
- **关键点完整性**: 完整度高的状态获得奖励分

### 5. 最终判定
取最高得分的状态为最终结果，置信度基于最高分与次高分的差距计算。

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 测试新算法
```bash
python test_new_algorithm.py
```

### 2. 运行演示
```bash
python demo.py
```

### 3. 处理图像（详细模式）
```bash
python main.py -i path/to/your/image.jpg -v
```

### 4. 处理视频
```bash
python main.py -i path/to/your/video.mp4 -o output_video.mp4
```

### 5. 实时摄像头检测
```bash
python main.py -c -v
```

### 6. 使用自定义模型
```bash
python main.py -i image.jpg -m yolo11l-pose.pt
```

## 🎮 交互控制

### 实时检测模式：
- **'q'键**: 退出程序
- **'v'键**: 切换详细信息显示

### 命令行参数：
- **-v, --verbose**: 显示详细分析信息
- **-i, --input**: 输入文件路径
- **-o, --output**: 输出文件路径
- **-m, --model**: YOLO模型路径
- **-c, --camera**: 使用摄像头

## 📊 状态识别

系统支持四种人体状态：

| 状态 | 中文 | 颜色 | 主要特征 |
|------|------|------|----------|
| Stand | 站立 | 绿色 | 身体垂直，腿部伸直 |
| Fall | 摔倒 | 红色 | 身体水平，宽高比大 |
| Sit | 坐着 | 紫色 | 腿部弯曲，特定角度 |
| other | 其他 | 黄色 | 关键点缺失或异常姿态 |

## 🔧 关键点说明

系统使用COCO格式的17个关键点：

```
0: nose (鼻子)          1: left_eye (左眼)      2: right_eye (右眼)
3: left_ear (左耳)      4: right_ear (右耳)     5: left_shoulder (左肩)
6: right_shoulder (右肩) 7: left_elbow (左肘)    8: right_elbow (右肘)
9: left_wrist (左腕)    10: right_wrist (右腕)  11: left_hip (左臀)
12: right_hip (右臀)    13: left_knee (左膝)    14: right_knee (右膝)
15: left_ankle (左踝)   16: right_ankle (右踝)
```

**核心关键点**（用于有效性检测）：
- 双肩: 5, 6
- 双髋: 11, 12  
- 双膝: 13, 14
- 双踝: 15, 16

## 📁 项目结构

```
YOLO-pose/
├── pose_detector.py         # 核心姿态检测类
├── main.py                 # 主程序入口
├── demo.py                 # 演示脚本
├── test_new_algorithm.py   # 新算法测试脚本
├── test_installation.py    # 安装测试脚本
├── requirements.txt        # 依赖包列表
└── README.md              # 项目说明
```

## 💻 使用示例

### Python代码示例

```python
from pose_detector import PoseDetector

# 初始化检测器
detector = PoseDetector('yolo11l-pose.pt')

# 处理图像
results = detector.process_image('test_image.jpg')

for result in results:
    analysis = result['analysis']
    print(f"状态: {analysis['status']}")
    print(f"置信度: {analysis['confidence']:.3f}")
    
    # 详细信息
    print("核心特征:")
    angles = analysis['angles']
    print(f"  身体角度: {angles['body_angle']:.1f}°")
    print(f"  宽高比: {angles['aspect_ratio']:.3f}")
    
    print("状态评分:")
    scores = analysis['scores']
    for status, score in scores.items():
        if score > 0:
            print(f"  {status}: {score:.2f}")
```

### 详细分析输出示例

```
=== 检测到的人 1 ===
最终状态: Stand
置信度: 0.856

关键点坐标:
  left_shoulder: (300, 150)
  right_shoulder: (340, 150)
  left_hip: (310, 250)
  right_hip: (330, 250)
  ...

身体中心点:
  shoulders_c: (320, 150)
  hips_c: (320, 250)
  knee_c: (320, 350)
  ankle_c: (322, 450)

核心特征:
  身体角度: 88.9°
  宽高比: 0.133
  肩髋垂直距离: 100.0
  髋膝肩角度: 170.5°
  踝膝髋角度: 175.2°

状态评分:
  Stand: 4.30
  Fall: 0.00
  Sit: 0.00
  other: 0.00

评分详情:
  无效关键点数量: 0
  body_angle_score: Stand +0.8 (接近垂直)
  aspect_ratio_score: Stand +0.6 (宽高比小)
  ankle_knee_hip_score: Stand +1.6 (腿部伸直)
  knee_position_score: Stand +0.4 (膝部高于肩部)
  hip_knee_shoulder_score: Stand +1.0 (身体挺直)
  vertical_diff_score: Stand +0.5 (肩髋距离大)
  completeness_bonus: Stand +0.3 (关键点完整)
```

## ⚙️ 性能优化建议

### 1. 模型选择
- **yolo11n-pose.pt**: 最快，适合实时应用
- **yolo11s-pose.pt**: 平衡速度和精度
- **yolo11m-pose.pt**: 更高精度
- **yolo11l-pose.pt**: 最高精度（推荐）
- **yolo12l-pose.pt**: 最新模型

### 2. 硬件加速
- 使用GPU可显著提升处理速度
- 确保安装了CUDA版本的PyTorch

### 3. 参数调整
- 调整置信度阈值以适应不同场景
- 根据实际需求修改评分权重
- 优化关键点有效性判断条件

## 🎯 应用场景

- 🏥 **医疗监护**: 老人跌倒检测，病人状态监控
- 🏭 **工业安全**: 工人安全监控，事故预防
- 🏃 **运动分析**: 运动姿态评估，动作纠正
- 🎮 **游戏娱乐**: 体感游戏控制，动作识别
- 📹 **视频分析**: 监控视频智能分析，行为识别
- 🏠 **智能家居**: 居家安全监控，老人看护

## ⚠️ 注意事项

1. 首次运行时会自动下载YOLO模型文件
2. 确保摄像头权限已开启（实时检测时）
3. 光照条件会影响检测精度
4. 遮挡严重时可能影响判断准确性
5. 建议在良好光照条件下使用
6. 多人场景下会分别分析每个人的姿态

## 🔄 算法更新日志

### v2.0 (当前版本)
- ✅ 新增坐着状态识别
- ✅ 改进关键点有效性检测
- ✅ 增加身体中心点计算
- ✅ 优化评分算法，支持多维度特征
- ✅ 详细的分析过程记录
- ✅ 更准确的置信度计算

### v1.0 (原版本)
- ✅ 基础站立/摔倒检测
- ✅ 简单的角度和比例分析

## 🤝 技术支持

如有问题或建议，请提交Issue或联系开发者。

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。 
