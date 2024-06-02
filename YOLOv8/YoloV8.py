import torch
from ultralytics import YOLO

def main():
    model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
    results = model.train(data='config.yaml', epochs=90, imgsz=850, batch=2)

if __name__ == '__main__':
    main()
