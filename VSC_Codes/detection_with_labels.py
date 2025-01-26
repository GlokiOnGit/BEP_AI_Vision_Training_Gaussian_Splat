import os
import cv2
from ultralytics import YOLO
import shutil

def main():

    # Load the trained YOLO model
    model = YOLO(r'C:\Users\20212127\Documents\Y4\BEP_AI_Vision_Training_Gaussian_Splat\AI_Vision_Trainings\ARBB.pt')

    # Define paths for test images, labels, and output
    source_directory = r'C:\Users\20212127\Documents\Y4\BEP_AI_Vision_Training_Gaussian_Splat\Validation_Dataset\images_orange_pink_match'
    label_directory = r'C:\Users\20212127\Documents\Y4\BEP_AI_Vision_Training_Gaussian_Splat\Validation_Dataset\labels_orange_pink_match'
    output_directory = r'C:\Users\20212127\Documents\Y4\results\output'
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
            results = model(image)
        
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
    metrics = model.val(data=r'C:\Users\20212127\Documents\Y4\results\test_data.yaml', cache=False)
    
    print(f"mAP (IoU=0.5): {metrics.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"Precision: {metrics.results_dict['metrics/precision(B)']:.3f}")
    print(f"Recall: {metrics.results_dict['metrics/recall(B)']:.3f}")
    print(f"F1 Score: {2 * (metrics.results_dict['metrics/precision(B)'] * metrics.results_dict['metrics/recall(B)']) / (metrics.results_dict['metrics/precision(B)'] + metrics.results_dict['metrics/recall(B)']):.3f}")
    print(f"mAP@0.50:0.95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")

    
    

if __name__ == '__main__':
    main()
