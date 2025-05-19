import os
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
import openvino as ov
from scipy.special import softmax
from transformers import CLIPProcessor, CLIPModel
import warnings
from visualize import visualize_result
warnings.filterwarnings('ignore')
# 禁用tokenizers并行以避免潜在问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def download_resources():
    """下载所需的资源文件"""
    # 下载可视化脚本
    urlretrieve(
        "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/228-clip-zero-shot-image-classification/visualize.py",
        filename='visualize.py'
    )
    
    
    # 下载示例图像
    sample_path = Path("data/fall.jpg")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    # if not sample_path.exists():
    #     urlretrieve(
    #         "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
    #         sample_path,
    #     )
    
    return visualize_result, sample_path

def load_model_and_processor():
    """加载预训练的CLIP模型和处理器"""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return model, processor

def prepare_inputs(image_path, labels, processor):
    """准备模型输入"""
    image = Image.open(image_path)
    text_descriptions = ["Someone falls in the picture", "No one falls in the picture"] #[f"This is a photo of a {label}" for label in input_labels]
    inputs = processor(text=text_descriptions, images=[image], return_tensors="pt", padding=True)
    return inputs, image

def run_pytorch_inference_cpu(model, inputs, n_runs=10):
    """运行PyTorch推理并计时（仅使用CPU）"""
    # 确保模型在CPU上
    model = model.cpu()
    
    # 预热
    with torch.no_grad():
        _ = model(**inputs)
    
    # 计时推理
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            results = model(**inputs)
    end_time = time.time()
    
    # 计算平均推理时间
    avg_time = (end_time - start_time) / n_runs
    
    # 获取结果
    logits_per_image = results['logits_per_image']
    probs = logits_per_image.softmax(dim=1).detach().numpy()
    
    return probs, avg_time

def run_pytorch_inference_cuda(model, inputs, n_runs=10):
    """运行PyTorch推理并计时（使用CUDA）"""
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("CUDA不可用，回退到CPU推理")
        return run_pytorch_inference_cpu(model, inputs, n_runs)
    
    # 将模型移动到GPU
    model = model.to("cuda")
    
    # 将输入数据移动到GPU
    cuda_inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # 预热
    with torch.no_grad():
        _ = model(**cuda_inputs)
    
    # 计时推理
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            results = model(**cuda_inputs)
    end_time = time.time()
    
    # 计算平均推理时间
    avg_time = (end_time - start_time) / n_runs
    
    # 获取结果并移回CPU
    logits_per_image = results['logits_per_image'].cpu()
    probs = logits_per_image.softmax(dim=1).numpy()
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    
    return probs, avg_time

def convert_to_openvino(model, inputs):
    """将PyTorch模型转换为OpenVINO IR格式"""
    # 确保模型在CPU上
    model = model.cpu()
    
    model.config.torchscript = True
    ov_model = ov.convert_model(model, example_input=dict(inputs))
    ov.save_model(ov_model, 'clip-vit-base-patch16.xml')
    return ov_model

def run_openvino_inference(ov_model, inputs, device='AUTO', n_runs=10):
    """运行OpenVINO推理并计时"""
    # 创建OpenVINO核心对象实例
    core = ov.Core()
    
    # 编译模型
    compiled_model = core.compile_model(ov_model, device)
    logits_per_image_out = compiled_model.output(0)
    
    # 预热
    _ = compiled_model(dict(inputs))[logits_per_image_out]
    
    # 计时推理
    start_time = time.time()
    for _ in range(n_runs):
        ov_logits_per_image = compiled_model(dict(inputs))[logits_per_image_out]
    end_time = time.time()
    
    # 计算平均推理时间
    avg_time = (end_time - start_time) / n_runs
    
    # 获取结果
    probs = softmax(ov_logits_per_image, axis=1)
    
    return probs, avg_time

def main():
    # 下载资源
    visualize_result, sample_path = download_resources()
    
    # 加载模型和处理器
    model, processor = load_model_and_processor()
    
    # 准备输入
    input_labels = ['cat', 'dog', 'wolf', 'tiger', 'man', 'horse', 'frog', 'tree', 'house', 'computer']
    inputs, image = prepare_inputs(sample_path, input_labels, processor) 
    
    # 运行PyTorch CPU推理
    print("运行PyTorch CPU推理...")
    pytorch_cpu_probs, pytorch_cpu_time = run_pytorch_inference_cpu(model, inputs)
    print(f"PyTorch CPU推理平均时间: {pytorch_cpu_time*1000:.2f} ms")
    print(f"PyTorch CPU概率: {pytorch_cpu_probs[0]}")
    
    # 运行PyTorch CUDA推理
    print("\n运行PyTorch CUDA推理...")
    pytorch_cuda_probs, pytorch_cuda_time = run_pytorch_inference_cuda(model, inputs)
    print(f"PyTorch CUDA推理平均时间: {pytorch_cuda_time*1000:.2f} ms")
    print(f"PyTorch CUDA概率: {pytorch_cuda_probs[0]}")
    
    # 可视化PyTorch CUDA结果
    visualize_result(image, ["Someone falls", "No one falls"], pytorch_cuda_probs[0])
    
    # 转换为OpenVINO模型
    print("\n转换为OpenVINO模型...")
    ov_model = convert_to_openvino(model, inputs)
    
    # 运行OpenVINO推理
    print("运行OpenVINO推理...")
    openvino_probs, openvino_time = run_openvino_inference(ov_model, inputs)
    print(f"OpenVINO推理平均时间: {openvino_time*1000:.2f} ms")
    print(f"OpenVINO概率: {openvino_probs[0]}")
    
    # 可视化OpenVINO结果
    visualize_result(image, ["Someone falls", "No one falls"], openvino_probs[0])
    
    # 比较性能
    print(f"\n性能比较:")
    if torch.cuda.is_available():
        cuda_cpu_speedup = pytorch_cpu_time / pytorch_cuda_time
        print(f"PyTorch CPU vs CUDA 加速比: {cuda_cpu_speedup:.2f}x")
    
    cpu_ov_speedup = pytorch_cpu_time / openvino_time
    print(f"PyTorch CPU vs OpenVINO 加速比: {cpu_ov_speedup:.2f}x")
    
    if torch.cuda.is_available():
        cuda_ov_speedup = pytorch_cuda_time / openvino_time
        print(f"PyTorch CUDA vs OpenVINO 加速比: {cuda_ov_speedup:.2f}x")
    
    # 比较结果准确性
    cpu_cuda_mae = np.mean(np.abs(pytorch_cpu_probs - pytorch_cuda_probs)) if torch.cuda.is_available() else 0
    cpu_ov_mae = np.mean(np.abs(pytorch_cpu_probs - openvino_probs))
    cuda_ov_mae = np.mean(np.abs(pytorch_cuda_probs - openvino_probs)) if torch.cuda.is_available() else 0
    
    print(f"\n结果准确性比较 (平均绝对误差):")
    if torch.cuda.is_available():
        print(f"PyTorch CPU vs CUDA: {cpu_cuda_mae:.6f}")
    print(f"PyTorch CPU vs OpenVINO: {cpu_ov_mae:.6f}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA vs OpenVINO: {cuda_ov_mae:.6f}")

if __name__ == "__main__":
    main()