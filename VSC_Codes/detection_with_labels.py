import os
import cv2
from ultralytics import YOLO
import shutil

def main():

    # Load the trained YOLO model
    model = YOLO(r'C:\Users\20212127\Documents\Y4\BEP\Training\trainings\ARBB.pt')

    # Define paths for test images, labels, and output
    source_directory = r'C:\Users\20212127\Documents\Y4\BEP\yolov8_training\dataset\detect_images_final_pink'
    label_directory = r'C:\Users\20212127\Documents\Y4\BEP\yolov8_training\dataset\detect_labels_final_pink'
    output_directory = r'C:\Users\20212127\Documents\Y4\BEP\yolov8_training\dataset\fdpink'
    os.makedirs(output_directory, exist_ok=True)

    # Process each image and evaluate
    for filename in os.listdir(source_directory):
        if filename.endswith('.jpeg'):
            image_path = os.path.join(source_directory, filename)
            print(f"Processing {image_path}...")

            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not open {filename}.")
                continue

            # Perform detection
            results = model(image, conf=0.8)
        
            # Load ground truth labels if available
            label_path = os.path.join(label_directory, filename.replace('.jpeg', '.txt'))
            if os.path.exists(label_path):
                with open(label_path, 'r') as file:
                    print(f"Loaded ground truth for {filename}.")
            else:
                print(f"No label file found for {filename}, skipping.")

            # Save annotated image
            output_path = os.path.join(output_directory, filename)
            annotated_image = results[0].plot()
            cv2.imwrite(output_path, annotated_image)
            print(f"Saved annotated image to {output_path}")

    # Run evaluation metrics
    metrics = model.val(data='C:/Users/20212127/Documents/Y4/BEP/yolov8_training/dataset/test_data.yaml', conf=0.8, cache=False)
    
    print(f"mAP (IoU=0.5): {metrics.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"Precision: {metrics.results_dict['metrics/precision(B)']:.3f}")
    print(f"Recall: {metrics.results_dict['metrics/recall(B)']:.3f}")
    print(f"F1 Score: {2 * (metrics.results_dict['metrics/precision(B)'] * metrics.results_dict['metrics/recall(B)']) / (metrics.results_dict['metrics/precision(B)'] + metrics.results_dict['metrics/recall(B)']):.3f}")
    

if __name__ == '__main__':
    main()
