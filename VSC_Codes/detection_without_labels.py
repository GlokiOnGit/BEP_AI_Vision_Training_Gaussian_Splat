import os
import cv2
from ultralytics import YOLO

def main():
    # Load the trained YOLO model
    model = YOLO(r'C:\Users\20212127\Documents\Y4\BEP\Training\trainings\GSR.pt')

    # Define paths for test images and output
    source_directory = r'C:\Users\20212127\Documents\Y4\BEP\yolov8_training\dataset\robot'
    output_directory = r'C:\Users\20212127\Documents\Y4\BEP\yolov8_training\dataset\output'
    os.makedirs(output_directory, exist_ok=True)

    # Process each image
    for filename in os.listdir(source_directory):
        if filename.endswith('.jpg'):
            image_path = os.path.join(source_directory, filename)
            print(f"Processing {image_path}...")

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not open {filename}. Skipping.")
                continue

            # Perform detection
            results = model(image, conf=0.8)
            
            # Check if detections were made
            if len(results[0].boxes) == 0:
                print(f"No objects detected in {filename}.")
                continue

            # Draw bounding boxes and save the annotated image
            annotated_image = results[0].plot()
            output_path = os.path.join(output_directory, filename)

            if not cv2.imwrite(output_path, annotated_image):
                print(f"Failed to save image to {output_path}.")
            else:
                print(f"Saved annotated image to {output_path}")

if __name__ == '__main__':
    main()
