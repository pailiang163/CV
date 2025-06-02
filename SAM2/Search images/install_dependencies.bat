@echo off
echo 正在安装图像相似度比较工具的依赖包...
echo.

echo 安装 OpenCV...
pip install opencv-python>=4.5.0

echo 安装 Pillow...
pip install Pillow>=8.0.0

echo 安装 imagehash...
pip install imagehash>=4.2.0

echo 安装 numpy...
pip install numpy>=1.19.0

echo 安装 scikit-image...
pip install scikit-image>=0.18.0

echo 安装 matplotlib...
pip install matplotlib>=3.3.0

echo.
echo 所有依赖包安装完成！
echo 现在可以运行图像相似度比较脚本了。
echo.
echo 推荐运行: python simple_angle_correction.py
pause 