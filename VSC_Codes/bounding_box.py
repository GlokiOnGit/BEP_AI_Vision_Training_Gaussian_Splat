import os
import cv2 # type: ignore
import matplotlib.pyplot as plt # type: ignore

def draw_bounding_boxes(image_folder, label_folder, output_folder):
    """
    Draw bounding boxes on images based on the YOLO format labels (.txt files).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all images in the folder
    for img_file in os.listdir(image_folder):
        if img_file.endswith('.jpeg'):
            # Load the image
            img_path = os.path.join(image_folder, img_file)
            img = cv2.imread(img_path)
            img_height, img_width, _ = img.shape

            # Corresponding label file
            label_file = img_file.replace('.jpeg', '.txt')
            label_path = os.path.join(label_folder, label_file)

            # Check if the label file exists
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                # Loop through each line in the label file (each bounding box)
                for line in lines:
                    parts = line.strip().split()
                    class_id = parts[0]
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Convert YOLO normalized coordinates to pixel coordinates
                    x_center_pixel = int(x_center * img_width)
                    y_center_pixel = int(y_center * img_height)
                    width_pixel = int(width * img_width)
                    height_pixel = int(height * img_height)

                    # Calculate the top-left corner of the bounding box
                    x_min = int(x_center_pixel - width_pixel / 2)
                    y_min = int(y_center_pixel - height_pixel / 2)
                    x_max = int(x_center_pixel + width_pixel / 2)
                    y_max = int(y_center_pixel + height_pixel / 2)

                    # Draw the bounding box
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Add class id text
                    cv2.putText(img, class_id, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save the image with bounding boxes
                output_path = os.path.join(output_folder, img_file)
                cv2.imwrite(output_path, img)
                print(f"Bounding boxes drawn and saved to {output_path}")

def show_image_with_boxes(image_path):
    """
    Display an image with bounding boxes using matplotlib.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')  # Hide axes for better visualization
    plt.show()

if __name__ == "__main__":
    # Paths to your folders
    image_folder = r'C:\Users\20212127\Documents\Y4\BEP\yolov8_training\dataset\detect_images_final_pink'  # Folder containing the images
    label_folder = r'C:\Users\20212127\Documents\Y4\BEP\yolov8_training\dataset\detect_labels_final_pink'  # Folder containing the labels
    output_folder = r'C:\\Users\\20212127\\Documents\\Y4\\BEP\\yolov8_training\\dataset\\output_box'  # Folder to save the images with drawn bounding boxes

    # Step 1: Draw bounding boxes and save the images
    draw_bounding_boxes(image_folder, label_folder, output_folder)

    # Step 2: Show a specific image with bounding boxes (Optional)
    # sample_image_path = os.path.join(output_folder, 'example_image.png')  # Change 'example_image.png' to one of your actual images
    # show_image_with_boxes(sample_image_path)