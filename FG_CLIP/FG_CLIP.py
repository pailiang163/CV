import torch
from PIL import Image
import time
import os
from pathlib import Path
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)

class ImageClassifier:
    def __init__(self, model_root="qihoo360/fg-clip-base", image_size=224):
        """
        初始化图像分类器
        
        Args:
            model_root: 模型路径
            image_size: 图像大小
        """
        self.model_root = model_root
        self.image_size = image_size
        
        # 加载模型和处理器
        print(f"正在加载模型 {model_root}...")
        start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True).cuda()
        self.device = self.model.device
        self.tokenizer = AutoTokenizer.from_pretrained(model_root)
        self.image_processor = AutoImageProcessor.from_pretrained(model_root)
        print(f"模型加载完成，耗时 {time.time() - start_time:.2f} 秒")
        # import openvino as ov
        # from PIL import Image
        # sample_path = Path("images/fall.jpg")
        # text_descriptions = ["Someone falls in the picture", "No one falls in the picture"]
        # image = Image.open(sample_path)

        # inputs = self.image_processor(text=text_descriptions, images=[image], return_tensors="pt", padding=True)
        # self.model.config.torchscript = True
        # self.ov_model = ov.convert_model(self.model, example_input=dict(inputs))
        # ov.save_model(self.ov_model, 'FG-clip-vit.xml')
        
    def preprocess_image(self, image_path):
        """
        预处理单张图片
        
        Args:
            image_path: 图片路径
        
        Returns:
            处理后的图片张量
        """
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image_input = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(self.device)
        return image_input
    
    def preprocess_images(self, image_paths):
        """
        批量预处理图片
        
        Args:
            image_paths: 图片路径列表
        
        Returns:
            处理后的图片张量批次
        """
        image_inputs = []
        for path in image_paths:
            image_inputs.append(self.preprocess_image(path))
        return torch.cat(image_inputs, dim=0)
    
    def preprocess_captions(self, captions, max_length=77, walk_short_pos=True):
        """
        预处理文本描述
        
        Args:
            captions: 文本描述列表
            max_length: 最大长度
            walk_short_pos: 短文本位置参数
        
        Returns:
            处理后的文本张量和walk_short_pos参数
        """
        caption_input = torch.tensor(
            self.tokenizer(captions, max_length=max_length, padding="max_length", truncation=True).input_ids, 
            dtype=torch.long, 
            device=self.device
        )
        return caption_input, walk_short_pos
    
    def predict(self, image_paths, captions, max_length=77, walk_short_pos=True):
        """
        预测图片与文本描述的匹配概率
        
        Args:
            image_paths: 图片路径或路径列表
            captions: 文本描述列表
            max_length: 最大文本长度
            walk_short_pos: 短文本位置参数
        
        Returns:
            预测结果字典，包含概率和推理时间
        """
        # 确保image_paths是列表
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        start_time = time.time()
        
        # 预处理图片
        image_inputs = self.preprocess_images(image_paths)
        
        # 预处理文本
        caption_input, walk_short_pos = self.preprocess_captions(captions, max_length, walk_short_pos)
        
        # 推理
        with torch.no_grad():
            # 获取图像特征
            image_feature = self.model.get_image_features(image_inputs)
            
            # 获取文本特征
            text_feature = self.model.get_text_features(caption_input, walk_short_pos=walk_short_pos)
            
            # 归一化特征
            image_feature = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
            text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)
            
            # 计算相似度
            logits_per_image = image_feature @ text_feature.T
            logits_per_image = self.model.logit_scale.exp() * logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        inference_time = time.time() - start_time
        
        # 构建结果
        results = []
        for i, path in enumerate(image_paths):
            results.append({
                "image_path": path,
                "image_name": os.path.basename(path),
                "probabilities": probs[i].cpu().numpy(),
                "predicted_class": int(torch.argmax(probs[i]).item()),
                "predicted_caption": captions[int(torch.argmax(probs[i]).item())]
            })
        
        return {
            "results": results,
            "inference_time": inference_time,
            "images_count": len(image_paths),
            "captions": captions
        }

# 使用示例
if __name__ == "__main__":
    # 初始化分类器（只需要加载一次模型）
    classifier = ImageClassifier(model_root="qihoo360/fg-clip-base", image_size=224)
    
    # 处理images文件夹下的所有图片
    images_folder = "images"
    image_paths = []
    
    # 获取文件夹中所有图片文件
    for filename in os.listdir(images_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_paths.append(os.path.join(images_folder, filename))
    
    if not image_paths:
        print(f"在 {images_folder} 文件夹中没有找到图片文件")
    else:
        print(f"找到 {len(image_paths)} 张图片")
        
        # 定义描述
        captions = ["Someone falls in the picture", "No one falls in the picture"]
        
        # 进行预测
        results = classifier.predict(image_paths, captions)
        
        # 打印结果
        print(f"\n推理总时间: {results['inference_time']:.4f} 秒")
        print(f"平均每张图片推理时间: {results['inference_time'] / len(image_paths):.4f} 秒")
        
        for i, result in enumerate(results['results']):
            print(f"\n图片 {i+1}: {result['image_name']}")
            print(f"预测类别: {result['predicted_caption']}")
            print(f"概率分布: {result['probabilities']}")