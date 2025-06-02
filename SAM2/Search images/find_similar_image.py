import os
import glob
from PIL import Image
import imagehash
import cv2
import numpy as np

# PyTorch相关导入 - 用于ResNet50和ViT特征提取
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    # 尝试导入ViT模型
    try:
        from transformers import ViTModel, ViTImageProcessor
        VIT_AVAILABLE = True
    except ImportError:
        VIT_AVAILABLE = False
        print("警告: transformers库未安装，ViT方法将不可用。请运行: pip install transformers")
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    VIT_AVAILABLE = False
    print("警告: PyTorch未安装，ResNet50和ViT方法将不可用")


def calculate_image_hash(img_path):
    """计算图片哈希值"""
    try:
        with Image.open(img_path) as img:
            return imagehash.dhash(img)
    except Exception as e:
        print(f"处理图片 {img_path} 时出错: {e}")
        return None


def hamming_distance(hash1, hash2):
    """计算汉明距离（越小越相似）"""
    if hash1 is None or hash2 is None:
        return float('inf')
    return hash1 - hash2


def calculate_histogram_similarity(img_path1, img_path2):
    """
    计算直方图相似度
    :param img_path1: 第一张图片路径
    :param img_path2: 第二张图片路径
    :return: 相似度分数 (0-1，越大越相似)
    """
    try:
        # 读取图片
        img1 = cv2.imread(img_path1)
        
        img2 = cv2.imread(img_path2)
       
        if img1 is None or img2 is None:
            return 0
        
        # 转换为HSV颜色空间
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

        
        # 计算直方图
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [100, 120, 120], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [100, 120, 120], [0, 180, 0, 256, 0, 256])
        
        # 归一化直方图
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # 计算相关性 (使用相关系数方法)
        # print("hist1: ", hist1)

        # print("hist2: ", hist2)

        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return max(0, correlation)  # 确保返回值在0-1之间
        
    except Exception as e:
        print(f"计算直方图相似度时出错 {img_path1} vs {img_path2}: {e}")
        return 0


def calculate_template_matching(img_path1, img_path2):
    """
    计算模板匹配相似度
    :param img_path1: 目标图片路径
    :param img_path2: 模板图片路径
    :return: 相似度分数 (0-1，越大越相似)
    """
    try:
        # 读取图片
        target = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        template = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        
        if target is None or template is None:
            return 0
        
        # 如果模板比目标图片大，交换它们
        if template.shape[0] > target.shape[0] or template.shape[1] > target.shape[1]:
            target, template = template, target
        
        # 执行模板匹配
        result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
        
        # 获取最大匹配值
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        return max(0, max_val)  # 确保返回值在0-1之间
        
    except Exception as e:
        print(f"计算模板匹配相似度时出错 {img_path1} vs {img_path2}: {e}")
        return 0


def calculate_structural_similarity(img_path1, img_path2):
    """
    计算结构相似度 (SSIM)
    :param img_path1: 第一张图片路径
    :param img_path2: 第二张图片路径
    :return: 相似度分数 (0-1，越大越相似)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        
        # 读取图片并转换为灰度
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return 0
        
        # 调整图片大小到相同尺寸
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        
        img1_resized = cv2.resize(img1, (width, height))
        img2_resized = cv2.resize(img2, (width, height))
        
        # 计算SSIM
        similarity = ssim(img1_resized, img2_resized)
        
        return max(0, similarity)
        
    except ImportError:
        print("警告: 未安装scikit-image，跳过SSIM计算")
        return 0
    except Exception as e:
        print(f"计算SSIM相似度时出错 {img_path1} vs {img_path2}: {e}")
        return 0


def calculate_sift_similarity(img_path1, img_path2):
    """
    使用SIFT特征计算图像相似度
    :param img_path1: 第一张图片路径
    :param img_path2: 第二张图片路径
    :return: 相似度分数 (0-1，越大越相似)
    """
    try:
        # 读取图片并转换为灰度
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return 0
        
        # 创建SIFT检测器
        sift = cv2.SIFT_create()
        
        # 检测关键点和描述符
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        # 如果没有检测到特征点，返回0
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return 0
        
        # 使用FLANN匹配器进行特征匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # 进行匹配
        matches = flann.knnMatch(des1, des2, k=2)
        
        # 应用Lowe's ratio test筛选好的匹配
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        # 计算相似度：好的匹配数量 / 最小特征点数量
        min_features = min(len(kp1), len(kp2))
        if min_features == 0:
            return 0
        
        similarity = len(good_matches) / min_features
        return min(1.0, similarity)  # 确保不超过1
        
    except Exception as e:
        print(f"计算SIFT相似度时出错 {img_path1} vs {img_path2}: {e}")
        return 0


def calculate_surf_similarity(img_path1, img_path2):
    """
    使用SURF特征计算图像相似度
    :param img_path1: 第一张图片路径
    :param img_path2: 第二张图片路径
    :return: 相似度分数 (0-1，越大越相似)
    """
    try:
        # 读取图片并转换为灰度
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return 0
        
        # 尝试创建SURF检测器（需要opencv-contrib-python）
        try:
            surf = cv2.xfeatures2d.SURF_create(400)
        except AttributeError:
            # 如果SURF不可用，使用ORB作为替代
            print("警告: SURF不可用，使用ORB特征作为替代")
            return calculate_orb_similarity(img_path1, img_path2)
        
        # 检测关键点和描述符
        kp1, des1 = surf.detectAndCompute(img1, None)
        kp2, des2 = surf.detectAndCompute(img2, None)
        
        # 如果没有检测到特征点，返回0
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return 0
        
        # 使用FLANN匹配器进行特征匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # 进行匹配
        matches = flann.knnMatch(des1, des2, k=2)
        
        # 应用Lowe's ratio test筛选好的匹配
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        # 计算相似度：好的匹配数量 / 最小特征点数量
        min_features = min(len(kp1), len(kp2))
        if min_features == 0:
            return 0
        
        similarity = len(good_matches) / min_features
        return min(1.0, similarity)  # 确保不超过1
        
    except Exception as e:
        print(f"计算SURF相似度时出错 {img_path1} vs {img_path2}: {e}")
        return 0


def calculate_orb_similarity(img_path1, img_path2):
    """
    使用ORB特征计算图像相似度（作为SURF的替代方案）
    :param img_path1: 第一张图片路径
    :param img_path2: 第二张图片路径
    :return: 相似度分数 (0-1，越大越相似)
    """
    try:
        # 读取图片并转换为灰度
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return 0
        
        # 创建ORB检测器
        orb = cv2.ORB_create(nfeatures=1000)
        
        # 检测关键点和描述符
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        # 如果没有检测到特征点，返回0
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return 0
        
        # 使用BFMatcher进行匹配（ORB使用汉明距离）
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # 按距离排序
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 筛选好的匹配（距离小于阈值）
        good_matches = [m for m in matches if m.distance < 50]
        
        # 计算相似度：好的匹配数量 / 最小特征点数量
        min_features = min(len(kp1), len(kp2))
        if min_features == 0:
            return 0
        
        similarity = len(good_matches) / min_features
        return min(1.0, similarity)  # 确保不超过1
        
    except Exception as e:
        print(f"计算ORB相似度时出错 {img_path1} vs {img_path2}: {e}")
        return 0


# ResNet50特征提取器（全局变量，避免重复加载模型）
_resnet_model = None
_resnet_transform = None


def get_resnet50_model():
    """
    获取预训练的ResNet50模型（单例模式）
    :return: ResNet50模型和预处理变换
    """
    global _resnet_model, _resnet_transform
    
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch未安装，无法使用ResNet50方法")
    
    if _resnet_model is None:
        print("正在加载预训练的ResNet50模型...")
        
        # 加载预训练的ResNet50模型
        _resnet_model = models.resnet50(pretrained=True)
        
        # 移除最后的分类层，只保留特征提取部分
        _resnet_model = nn.Sequential(*list(_resnet_model.children())[:-1])
        
        # 设置为评估模式
        _resnet_model.eval()
        
        # 定义图像预处理变换
        _resnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet50输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])
        
        print("ResNet50模型加载完成")
    
    return _resnet_model, _resnet_transform


def extract_resnet50_features(img_path):
    """
    使用ResNet50提取图像特征
    :param img_path: 图片路径
    :return: 特征向量 (numpy array) 或 None
    """
    try:
        if not PYTORCH_AVAILABLE:
            return None
        
        # 获取模型和预处理变换
        model, transform = get_resnet50_model()
        
        # 读取并预处理图像
        with Image.open(img_path) as img:
            # 转换为RGB（如果是灰度图或RGBA）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 应用预处理变换
            img_tensor = transform(img).unsqueeze(0)  # 添加batch维度
            
            # 提取特征
            with torch.no_grad():
                features = model(img_tensor)
                # 展平特征向量
                features = features.view(features.size(0), -1)
                # 转换为numpy数组
                features = features.numpy().flatten()
                
                # L2归一化
                features = features / np.linalg.norm(features)
                
                return features
    
    except Exception as e:
        print(f"提取ResNet50特征时出错 {img_path}: {e}")
        return None


def calculate_resnet50_similarity(img_path1, img_path2):
    """
    使用ResNet50特征计算图像相似度
    :param img_path1: 第一张图片路径
    :param img_path2: 第二张图片路径
    :return: 相似度分数 (0-1，越大越相似)
    """
    try:
        if not PYTORCH_AVAILABLE:
            print("警告: PyTorch未安装，跳过ResNet50计算")
            return 0
        
        # 提取两张图片的特征
        features1 = extract_resnet50_features(img_path1)
        features2 = extract_resnet50_features(img_path2)
        
        if features1 is None or features2 is None:
            return 0
        
        # 计算余弦相似度
        cosine_similarity = np.dot(features1, features2)
        
        # 确保相似度在0-1之间
        similarity = max(0, min(1, cosine_similarity )) #(cosine_similarity + 1) / 2)
        
        return similarity
        
    except Exception as e:
        print(f"计算ResNet50相似度时出错 {img_path1} vs {img_path2}: {e}")
        return 0


def calculate_resnet50_similarity_batch(target_image, image_list):
    """
    批量计算ResNet50相似度（优化版本）
    :param target_image: 目标图片路径
    :param image_list: 待比较的图片路径列表
    :return: 相似度分数列表
    """
    try:
        if not PYTORCH_AVAILABLE:
            return [0] * len(image_list)
        
        # 提取目标图片特征
        target_features = extract_resnet50_features(target_image)
        if target_features is None:
            return [0] * len(image_list)
        
        similarities = []
        
        for img_path in image_list:
            # 跳过目标图片本身
            if os.path.abspath(img_path) == os.path.abspath(target_image):
                similarities.append(0)
                continue
            
            # 提取当前图片特征
            features = extract_resnet50_features(img_path)
            
            if features is None:
                similarities.append(0)
                continue
            
            # 计算余弦相似度
            cosine_similarity = np.dot(target_features, features)
            similarity = max(0, min(1, (cosine_similarity + 1) / 2))
            similarities.append(similarity)
        
        return similarities
        
    except Exception as e:
        print(f"批量计算ResNet50相似度时出错: {e}")
        return [0] * len(image_list)


# ViT特征提取器（全局变量，避免重复加载模型）
_vit_model = None
_vit_processor = None


def get_vit_model():
    """
    获取预训练的ViT模型（单例模式）
    :return: ViT模型和图像处理器
    """
    global _vit_model, _vit_processor
    
    if not VIT_AVAILABLE:
        raise ImportError("transformers库未安装，无法使用ViT方法")
    
    if _vit_model is None:
        print("正在加载预训练的ViT模型...")
        
        # 使用google/vit-base-patch16-224-in21k预训练模型
        model_name = "google/vit-base-patch16-224-in21k"
        
        try:
            # 加载预训练的ViT模型和处理器
            _vit_model = ViTModel.from_pretrained(model_name)
            _vit_processor = ViTImageProcessor.from_pretrained(model_name)
            
            # 设置为评估模式
            _vit_model.eval()
            
            print("ViT模型加载完成")
            
        except Exception as e:
            print(f"加载ViT模型失败: {e}")
            print("尝试使用本地缓存或检查网络连接...")
            raise
    
    return _vit_model, _vit_processor


def extract_vit_features(img_path):
    """
    使用ViT提取图像特征
    :param img_path: 图片路径
    :return: 特征向量 (numpy array) 或 None
    """
    try:
        if not VIT_AVAILABLE:
            return None
        
        # 获取模型和处理器
        model, processor = get_vit_model()
        
        # 读取并预处理图像
        with Image.open(img_path) as img:
            # 转换为RGB（如果是灰度图或RGBA）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 使用ViT处理器预处理图像
            inputs = processor(images=img, return_tensors="pt")
            
            # 提取特征
            with torch.no_grad():
                outputs = model(**inputs)
                # 使用[CLS] token的特征作为图像表示
                features = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                
                # L2归一化
                features = features / np.linalg.norm(features)
                
                return features
    
    except Exception as e:
        print(f"提取ViT特征时出错 {img_path}: {e}")
        return None


def calculate_vit_similarity(img_path1, img_path2):
    """
    使用ViT特征计算图像相似度
    :param img_path1: 第一张图片路径
    :param img_path2: 第二张图片路径
    :return: 相似度分数 (0-1，越大越相似)
    """
    try:
        if not VIT_AVAILABLE:
            print("警告: transformers库未安装，跳过ViT计算")
            return 0
        
        # 提取两张图片的特征
        features1 = extract_vit_features(img_path1)
        features2 = extract_vit_features(img_path2)
        
        if features1 is None or features2 is None:
            return 0
        
        # 计算余弦相似度
        cosine_similarity = np.dot(features1, features2)
        
        # 确保相似度在0-1之间
        similarity = max(0, min(1, cosine_similarity))
        
        return similarity
        
    except Exception as e:
        print(f"计算ViT相似度时出错 {img_path1} vs {img_path2}: {e}")
        return 0


def calculate_vit_similarity_batch(target_image, image_list):
    """
    批量计算ViT相似度（优化版本）
    :param target_image: 目标图片路径
    :param image_list: 待比较的图片路径列表
    :return: 相似度分数列表
    """
    try:
        if not VIT_AVAILABLE:
            return [0] * len(image_list)
        
        # 提取目标图片特征
        target_features = extract_vit_features(target_image)
        if target_features is None:
            return [0] * len(image_list)
        
        similarities = []
        
        for img_path in image_list:
            # 跳过目标图片本身
            if os.path.abspath(img_path) == os.path.abspath(target_image):
                similarities.append(0)
                continue
            
            # 提取当前图片特征
            features = extract_vit_features(img_path)
            
            if features is None:
                similarities.append(0)
                continue
            
            # 计算余弦相似度
            cosine_similarity = np.dot(target_features, features)
            similarity = max(0, min(1, cosine_similarity))
            similarities.append(similarity)
        
        return similarities
        
    except Exception as e:
        print(f"批量计算ViT相似度时出错: {e}")
        return [0] * len(image_list)


def get_all_images(folder_path):
    """获取文件夹中所有图片文件"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']
    image_files = []
    
    for ext in image_extensions:
        # 搜索当前目录
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))
        # 递归搜索子目录
        pattern = os.path.join(folder_path, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
    
    return sorted(list(set(image_files)))


def get_subfolders(folder_path):
    """获取文件夹中的所有子文件夹"""
    try:
        subfolders = []
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                subfolders.append(item_path)
        return sorted(subfolders)
    except Exception as e:
        print(f"获取子文件夹时出错: {e}")
        return []


def find_most_similar_image_in_folder(target_image, search_folder, method='hash'):
    """
    在单个文件夹中找到最相似的图片
    :param target_image: 目标图片路径
    :param search_folder: 搜索文件夹路径
    :param method: 相似度计算方法 ('hash', 'histogram', 'template', 'ssim', 'sift', 'surf', 'orb', 'resnet50', 'combined')
    """
    # 获取所有图片文件
    image_files = get_all_images(search_folder)
    
    if not image_files:
        return None
    
    similarities = []
    processed = 0
    
    # 如果使用哈希方法，先计算目标图片的哈希值
    target_hash = None
    if method in ['hash', 'combined']:
        target_hash = calculate_image_hash(target_image)
        if target_hash is None and method == 'hash':
            return None
    
    for img_path in image_files:
        # 跳过目标图片本身
        if os.path.abspath(img_path) == os.path.abspath(target_image):
            continue
        
        similarity_score = 0
        distance = float('inf')
        
        if method == 'hash':
            # 哈希方法
            img_hash = calculate_image_hash(img_path)
            if img_hash is not None and target_hash is not None:
                distance = hamming_distance(target_hash, img_hash)
                similarity_score = max(0, (64 - distance) / 64 * 100)
        
        elif method == 'histogram':
            # 直方图方法
            hist_similarity = calculate_histogram_similarity(target_image, img_path)
            similarity_score = hist_similarity * 100
            distance = 1 - hist_similarity  # 转换为距离（越小越相似）
        
        elif method == 'template':
            # 模板匹配方法
            template_similarity = calculate_template_matching(target_image, img_path)
            similarity_score = template_similarity * 100
            distance = 1 - template_similarity
        
        elif method == 'ssim':
            # 结构相似度方法
            ssim_similarity = calculate_structural_similarity(target_image, img_path)
            similarity_score = ssim_similarity * 100
            distance = 1 - ssim_similarity
        
        elif method == 'sift':
            # SIFT特征方法
            sift_similarity = calculate_sift_similarity(target_image, img_path)
            similarity_score = sift_similarity * 100
            distance = 1 - sift_similarity
        
        elif method == 'surf':
            # SURF特征方法
            surf_similarity = calculate_surf_similarity(target_image, img_path)
            similarity_score = surf_similarity * 100
            distance = 1 - surf_similarity
        
        elif method == 'orb':
            # ORB特征方法
            orb_similarity = calculate_orb_similarity(target_image, img_path)
            similarity_score = orb_similarity * 100
            distance = 1 - orb_similarity
        
        elif method == 'resnet50':
            # ResNet50深度学习特征方法
            resnet_similarity = calculate_resnet50_similarity(target_image, img_path)
            similarity_score = resnet_similarity * 100
            distance = 1 - resnet_similarity
        
        elif method == 'vit':
            # ViT (Vision Transformer) 深度学习特征方法
            vit_similarity = calculate_vit_similarity(target_image, img_path)
            similarity_score = vit_similarity * 100
            distance = 1 - vit_similarity
        
        elif method == 'combined':
            # 综合方法：结合多种算法
            scores = []
            weights = []
            
            # 哈希相似度
            if target_hash is not None:
                img_hash = calculate_image_hash(img_path)
                if img_hash is not None:
                    hash_distance = hamming_distance(target_hash, img_hash)
                    hash_score = max(0, (64 - hash_distance) / 64)
                    scores.append(hash_score)
                    weights.append(0.1)  # 权重10%
            
            # 直方图相似度
            hist_score = calculate_histogram_similarity(target_image, img_path)
            if hist_score > 0:
                scores.append(hist_score)
                weights.append(0.1)  # 权重10%
            
            # 模板匹配相似度
            template_score = calculate_template_matching(target_image, img_path)
            if template_score > 0:
                scores.append(template_score)
                weights.append(0.05)  # 权重5%
            
            # SSIM相似度
            ssim_score = calculate_structural_similarity(target_image, img_path)
            if ssim_score > 0:
                scores.append(ssim_score)
                weights.append(0.1)  # 权重10%
            
            # SIFT特征相似度
            sift_score = calculate_sift_similarity(target_image, img_path)
            if sift_score > 0:
                scores.append(sift_score)
                weights.append(0.15)  # 权重15%
            
            # ResNet50深度学习特征相似度
            if PYTORCH_AVAILABLE:
                resnet_score = calculate_resnet50_similarity(target_image, img_path)
                if resnet_score > 0:
                    scores.append(resnet_score)
                    weights.append(0.25)  # 权重25%
            
            # ViT深度学习特征相似度
            if VIT_AVAILABLE:
                vit_score = calculate_vit_similarity(target_image, img_path)
                if vit_score > 0:
                    scores.append(vit_score)
                    weights.append(0.25)  # 权重25%
            
            # 计算加权平均
            if scores and weights:
                # 归一化权重
                total_weight = sum(weights[:len(scores)])
                normalized_weights = [w/total_weight for w in weights[:len(scores)]]
                
                weighted_score = sum(s * w for s, w in zip(scores, normalized_weights))
                similarity_score = weighted_score * 100
                distance = 1 - weighted_score
            else:
                similarity_score = 0
                distance = float('inf')
        
        if similarity_score > 0:
            similarities.append((img_path, distance, similarity_score))
        
        processed += 1
    
    if not similarities:
        return None
    
    # 按距离排序（距离越小越相似）
    similarities.sort(key=lambda x: x[1])
    print("相似前十名：")
    print(similarities[0:5])
    
    return similarities[0]  # 返回最相似的图片


def find_most_similar_image_by_folders(target_image, base_folder, method='hash'):
    """
    对每个子文件夹分别计算相似度
    :param target_image: 目标图片路径
    :param base_folder: 基础文件夹路径 (如 opt_result)
    :param method: 相似度计算方法
    """
    print(f" 目标图片: {target_image}")
    print(f" 基础文件夹: {base_folder}")
    print(f" 计算方法: {method}")
    
    # 检查文件是否存在
    if not os.path.exists(target_image):
        print(f" 目标图片不存在: {target_image}")
        return None
    
    if not os.path.exists(base_folder):
        print(f" 基础文件夹不存在: {base_folder}")
        return None
    
    # 获取所有子文件夹
    subfolders = get_subfolders(base_folder)
    
    if not subfolders:
        print(f" 在 {base_folder} 中未找到子文件夹")
        return None
    
    print(f" 找到 {len(subfolders)} 个子文件夹，开始分别计算相似度...")
    
    folder_results = []
    
    for i, subfolder in enumerate(subfolders, 1):
        folder_name = os.path.basename(subfolder)
        print(f"\n{'='*60}")
        print(f"处理文件夹 {i}/{len(subfolders)}: {folder_name}")
        print(f"{'='*60}")
        
        # 获取该文件夹中的图片数量
        images_in_folder = get_all_images(subfolder)
        print(f"文件夹中有 {len(images_in_folder)} 张图片")
        
        if not images_in_folder:
            print("跳过: 文件夹中没有图片")
            continue
        
        # 在该文件夹中找最相似的图片
        best_match = find_most_similar_image_in_folder(target_image, subfolder, method)
        
        if best_match:
            img_path, distance, similarity_score = best_match
            rel_path = os.path.relpath(img_path)
            
            print(f"最相似图片: {os.path.basename(img_path)}")
            if method == 'hash':
                print(f"汉明距离: {int(distance)}")
            else:
                print(f"差异度: {distance:.4f}")
            print(f"相似度: {similarity_score:.1f}%")
            
            folder_results.append((subfolder, img_path, distance, similarity_score))
        else:
            print("未找到相似的图片")
    
    if not folder_results:
        print("\n 所有文件夹都没有找到相似的图片")
        return None
    
    # 按相似度排序所有文件夹的结果
    folder_results.sort(key=lambda x: x[2])  # 按距离排序（越小越相似）
    
    # 显示总体结果
    print(f"\n{'='*100}")
    print(f" 各文件夹相似度排行榜 - 方法: {method.upper()}")
    print(f"{'='*100}")
    print(f"{'排名':<4} {'文件夹名':<35} {'距离/差异':<12} {'相似度':<10} {'最相似图片'}")
    print("-" * 100)
    
    for i, (subfolder, img_path, distance, similarity) in enumerate(folder_results, 1):
        folder_name = os.path.basename(subfolder)
        img_name = os.path.basename(img_path)
        
        if method == 'hash':
            distance_str = f"{int(distance)}"
        else:
            distance_str = f"{distance:.4f}"
        
        print(f"{i:<4} {folder_name:<35} {distance_str:<12} {similarity:.1f}%{'':<6} {img_name}")
    
    # 显示最佳匹配
    best_folder, best_img, best_distance, best_similarity = folder_results[0]
    print(f"\n 最佳匹配:")
    print(f"   文件夹: {os.path.basename(best_folder)}")
    print(f"    图片: {os.path.basename(best_img)}")
    print(f"   完整路径: {os.path.relpath(best_img)}")
    if method == 'hash':
        print(f"   汉明距离: {int(best_distance)}")
    else:
        print(f"   差异度: {best_distance:.4f}")
    print(f"   相似度: {best_similarity:.1f}%")
    
    if method == 'hash':
        if best_distance == 0:
            print("   完全相同的图片！")
        elif best_distance <= 5:
            print("   非常相似！")
        elif best_distance <= 10:
            print("   比较相似")
        else:
            print("   相似度较低")
    else:
        if best_similarity >= 90:
            print("   非常相似！")
        elif best_similarity >= 70:
            print("   比较相似")
        elif best_similarity >= 50:
            print("   有一定相似性")
        else:
            print("   相似度较低")
    
    return folder_results


def find_most_similar_image(target_image, search_folder, method='hash'):
    """
    找到最相似的图片（原有功能保持不变）
    :param target_image: 目标图片路径
    :param search_folder: 搜索文件夹路径
    :param method: 相似度计算方法 ('hash', 'histogram', 'template', 'ssim', 'sift', 'surf', 'orb', 'resnet50', 'combined')
    """
    print(f" 目标图片: {target_image}")
    print(f" 搜索文件夹: {search_folder}")
    print(f" 计算方法: {method}")
    
    # 检查文件是否存在
    if not os.path.exists(target_image):
        print(f" 目标图片不存在: {target_image}")
        return None
    
    if not os.path.exists(search_folder):
        print(f" 搜索文件夹不存在: {search_folder}")
        return None
    
    # 获取所有图片文件
    print(" 正在搜索图片文件...")
    image_files = get_all_images(search_folder)
    
    if not image_files:
        print(" 未找到任何图片文件")
        return None
    
    print(f" 找到 {len(image_files)} 张图片，开始比较相似度...")
    
    similarities = []
    processed = 0
    
    # 如果使用哈希方法，先计算目标图片的哈希值
    target_hash = None
    if method in ['hash', 'combined']:
        print(" 正在计算目标图片哈希值...")
        target_hash = calculate_image_hash(target_image)
        if target_hash is None and method == 'hash':
            print(" 无法处理目标图片")
            return None
    
    for img_path in image_files:
        # 跳过目标图片本身
        if os.path.abspath(img_path) == os.path.abspath(target_image):
            continue
        
        similarity_score = 0
        distance = float('inf')
        
        if method == 'hash':
            # 哈希方法
            img_hash = calculate_image_hash(img_path)
            if img_hash is not None and target_hash is not None:
                distance = hamming_distance(target_hash, img_hash)
                similarity_score = max(0, (64 - distance) / 64 * 100)
        
        elif method == 'histogram':
            # 直方图方法
            hist_similarity = calculate_histogram_similarity(target_image, img_path)
            similarity_score = hist_similarity * 100
            distance = 1 - hist_similarity  # 转换为距离（越小越相似）
        
        elif method == 'template':
            # 模板匹配方法
            template_similarity = calculate_template_matching(target_image, img_path)
            similarity_score = template_similarity * 100
            distance = 1 - template_similarity
        
        elif method == 'ssim':
            # 结构相似度方法
            ssim_similarity = calculate_structural_similarity(target_image, img_path)
            similarity_score = ssim_similarity * 100
            distance = 1 - ssim_similarity
        
        elif method == 'sift':
            # SIFT特征方法
            sift_similarity = calculate_sift_similarity(target_image, img_path)
            similarity_score = sift_similarity * 100
            distance = 1 - sift_similarity
        
        elif method == 'surf':
            # SURF特征方法
            surf_similarity = calculate_surf_similarity(target_image, img_path)
            similarity_score = surf_similarity * 100
            distance = 1 - surf_similarity
        
        elif method == 'orb':
            # ORB特征方法
            orb_similarity = calculate_orb_similarity(target_image, img_path)
            similarity_score = orb_similarity * 100
            distance = 1 - orb_similarity
        
        elif method == 'resnet50':
            # ResNet50深度学习特征方法
            resnet_similarity = calculate_resnet50_similarity(target_image, img_path)
            similarity_score = resnet_similarity * 100
            distance = 1 - resnet_similarity
        
        elif method == 'vit':
            # ViT (Vision Transformer) 深度学习特征方法
            vit_similarity = calculate_vit_similarity(target_image, img_path)
            similarity_score = vit_similarity * 100
            distance = 1 - vit_similarity
        
        elif method == 'combined':
            # 综合方法：结合多种算法
            scores = []
            weights = []
            
            # 哈希相似度
            if target_hash is not None:
                img_hash = calculate_image_hash(img_path)
                if img_hash is not None:
                    hash_distance = hamming_distance(target_hash, img_hash)
                    hash_score = max(0, (64 - hash_distance) / 64)
                    scores.append(hash_score)
                    weights.append(0.1)  # 权重10%
            
            # 直方图相似度
            hist_score = calculate_histogram_similarity(target_image, img_path)
            if hist_score > 0:
                scores.append(hist_score)
                weights.append(0.1)  # 权重10%
            
            # 模板匹配相似度
            template_score = calculate_template_matching(target_image, img_path)
            if template_score > 0:
                scores.append(template_score)
                weights.append(0.05)  # 权重5%
            
            # SSIM相似度
            ssim_score = calculate_structural_similarity(target_image, img_path)
            if ssim_score > 0:
                scores.append(ssim_score)
                weights.append(0.1)  # 权重10%
            
            # SIFT特征相似度
            sift_score = calculate_sift_similarity(target_image, img_path)
            if sift_score > 0:
                scores.append(sift_score)
                weights.append(0.15)  # 权重15%
            
            # ResNet50深度学习特征相似度
            if PYTORCH_AVAILABLE:
                resnet_score = calculate_resnet50_similarity(target_image, img_path)
                if resnet_score > 0:
                    scores.append(resnet_score)
                    weights.append(0.25)  # 权重25%
            
            # ViT深度学习特征相似度
            if VIT_AVAILABLE:
                vit_score = calculate_vit_similarity(target_image, img_path)
                if vit_score > 0:
                    scores.append(vit_score)
                    weights.append(0.25)  # 权重25%
            
            # 计算加权平均
            if scores and weights:
                # 归一化权重
                total_weight = sum(weights[:len(scores)])
                normalized_weights = [w/total_weight for w in weights[:len(scores)]]
                
                weighted_score = sum(s * w for s, w in zip(scores, normalized_weights))
                similarity_score = weighted_score * 100
                distance = 1 - weighted_score
            else:
                similarity_score = 0
                distance = float('inf')
        
        if similarity_score > 0:
            similarities.append((img_path, distance, similarity_score))
        
        processed += 1
        if processed % 20 == 0 or processed == len(image_files):
            print(f"   进度: {processed}/{len(image_files)} ({processed/len(image_files)*100:.1f}%)")
    
    if not similarities:
        print(" 没有成功处理的图片")
        return None
    
    # 按距离排序（距离越小越相似）
    similarities.sort(key=lambda x: x[1])
    
    # 显示结果
    print(f"\n{'='*80}")
    print(f" 相似度排行榜 (前10名) - 方法: {method}")
    print(f"{'='*80}")
    print(f"{'排名':<4} {'距离/差异':<12} {'相似度':<10} {'图片路径'}")
    print("-" * 80)
    
    top_10 = similarities[:10]
    for i, (img_path, distance, similarity) in enumerate(top_10, 1):
        rel_path = os.path.relpath(img_path)
        if method == 'hash':
            distance_str = f"{int(distance)}"
        else:
            distance_str = f"{distance:.4f}"
        print(f"{i:<4} {distance_str:<12} {similarity:.1f}%{'':<6} {rel_path}")
    
    # 显示最相似的图片
    best_match = similarities[0]
    print(f"\n 最相似的图片:")
    print(f"    路径: {os.path.relpath(best_match[0])}")
    if method == 'hash':
        print(f"    汉明距离: {int(best_match[1])}")
    else:
        print(f"    差异度: {best_match[1]:.4f}")
    print(f"    相似度: {best_match[2]:.1f}%")
    
    if method == 'hash':
        if best_match[1] == 0:
            print("    完全相同的图片！")
        elif best_match[1] <= 5:
            print("    非常相似！")
        elif best_match[1] <= 10:
            print("    比较相似")
        else:
            print("    相似度较低")
    else:
        if best_match[2] >= 90:
            print("    非常相似！")
        elif best_match[2] >= 70:
            print("    比较相似")
        elif best_match[2] >= 50:
            print("    有一定相似性")
        else:
            print("    相似度较低")
    
    return best_match


if __name__ == '__main__':
    # 配置参数 - 您可以根据需要修改这些路径
    target_image = 'sample1.jpg'
    
    # 基础文件夹路径
    base_folder = 'opt_result'
    
    # 选择相似度计算方法:
    # 'hash' - 感知哈希 (快速，适合检测相同或非常相似的图片)
    # 'histogram' - 直方图比较 (适合检测颜色分布相似的图片)
    # 'template' - 模板匹配 (适合检测结构相似的图片)
    # 'ssim' - 结构相似度 (需要安装scikit-image: pip install scikit-image)
    # 'sift' - SIFT特征匹配 (对旋转、缩放、光照变化鲁棒，适合检测相同物体)
    # 'surf' - SURF特征匹配 (比SIFT更快，需要opencv-contrib-python)
    # 'orb' - ORB特征匹配 (免费替代SIFT/SURF，速度快)
    # 'resnet50' - ResNet50深度学习特征匹配
    # 'vit' - ViT深度学习特征匹配
    # 'combined' - 综合方法 (结合多种算法，更准确但较慢)
    
    methods = ["vit"]
    
    for method in methods:
        print(f"\n{'='*100}")
        print(f" 使用 {method.upper()} 方法对各文件夹分别进行图像相似度分析...")
        print(f"{'='*100}")
        
        # 对每个子文件夹分别计算相似度
        result = find_most_similar_image_by_folders(target_image, base_folder, method=method)
        
        if result:
            print(f"\n  分析完成！在 {len(result)} 个文件夹中找到了相似图片")
        else:
            print(f"\n  {method} 方法分析失败")
        
        print("\n" + "="*50 + " 方法完成 " + "="*50) 


        print("------------------------------------------------------")
        #find_most_similar_image(target_image, base_folder, method=method)