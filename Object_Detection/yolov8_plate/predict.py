
from ultralytics import YOLO

if __name__ == '__main__':
        # Load a model
        
        model = YOLO("/home/featurize/work/2025/CV/CV/Object_Detection/yolov8_plate/runs/detect/train5/weights/best.pt")  # load model
        model.predict(source="/home/featurize/work/2024/20220917/datasets/plate_images/images", save=True, save_conf=True, save_txt=True, name='output')
                
