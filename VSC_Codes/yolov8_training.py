import os
from ultralytics import YOLO

def main():
    # Set up working directory
    os.chdir(r"C:\Users\20212127\Documents\Y4\BEP\yolov8_training\dataset")

    # Load YOLO model for training
    model = YOLO('yolov8s.pt')

    # Train the model with specified parameters
    results = model.train(data='data.yaml', epochs=10, imgsz=640, batch=16, amp=False, device="cuda:0")

    print("Training completed.")

if __name__ == "__main__":
    main()
