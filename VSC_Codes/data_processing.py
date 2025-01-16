import os
import random
import shutil
from PIL import Image

def setup_dataset(images_folder, labels_folder, save_folder, val_ratio=0.2):
    """
    Sets up the dataset by splitting it into training and validation sets.
    Removes invalid or empty labels, then sorts valid files into the correct folders.
    """
    # Define paths for the dataset structure
    train_images_dir = os.path.join(save_folder, 'images', 'train')
    val_images_dir = os.path.join(save_folder, 'images', 'val')
    train_labels_dir = os.path.join(save_folder, 'labels', 'train')
    val_labels_dir = os.path.join(save_folder, 'labels', 'val')

    # Create required directories
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    print("Directories created successfully.")

    # Collect image and label file names
    images = [f for f in os.listdir(images_folder) if f.endswith('.png')]
    labels = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]

    # Ensure each image has a corresponding label file
    image_label_pairs = [(img, img.replace('.png', '.txt')) for img in images if img.replace('.png', '.txt') in labels]

    # Remove invalid or empty labels and their images
    valid_image_label_pairs = []
    for img, lbl in image_label_pairs:
        img_path = os.path.join(images_folder, img)
        lbl_path = os.path.join(labels_folder, lbl)

        if is_valid_label(img_path, lbl_path):
            valid_image_label_pairs.append((img, lbl))
        else:
            print(f"Removing invalid pair: {img}, {lbl}")
            os.remove(img_path)
            os.remove(lbl_path)

    # Shuffle and split images into training and validation sets
    random.shuffle(valid_image_label_pairs)
    split_index = int(len(valid_image_label_pairs) * (1 - val_ratio))
    train_pairs = valid_image_label_pairs[:split_index]
    val_pairs = valid_image_label_pairs[split_index:]

    # Copy files to train/val folders
    for img, lbl in train_pairs:
        shutil.copy(os.path.join(images_folder, img), os.path.join(train_images_dir, img))
        shutil.copy(os.path.join(labels_folder, lbl), os.path.join(train_labels_dir, lbl))

    for img, lbl in val_pairs:
        shutil.copy(os.path.join(images_folder, img), os.path.join(val_images_dir, img))
        shutil.copy(os.path.join(labels_folder, lbl), os.path.join(val_labels_dir, lbl))

    print("Dataset setup completed.")

    # Create data.yaml for training configuration
    create_yaml_file(save_folder)

def is_valid_label(image_path, label_path):
    """
    Checks if a label file is valid: not empty and all coordinates are between 0 and 1.
    """
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:  # Empty label file
            return False

        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                return False

        return True
    except Exception as e:
        print(f"Error validating label {label_path}: {e}")
        return False

def create_yaml_file(save_folder):
    """
    Generates data.yaml file needed for YOLOv8 training.
    """
    yaml_path = os.path.join(save_folder, 'data.yaml')
    yaml_content = f"""
path: {save_folder}  # Root directory for dataset
train: images/train
val: images/val
nc: 1
names: ['ball']
"""

    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(yaml_content.strip())
    print(f"data.yaml file created at {yaml_path}")

if __name__ == "__main__":
    images_folder = r'C:\Users\20212127\Documents\Y4\BEP\yolov8_training\images'
    labels_folder = r'C:\Users\20212127\Documents\Y4\BEP\yolov8_training\labels'
    save_folder = r'C:\Users\20212127\Documents\Y4\BEP\yolov8_training\dataset'
    setup_dataset(images_folder, labels_folder, save_folder)
