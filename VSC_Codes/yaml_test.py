import os

def create_test_yaml_file(save_folder):
    """
    Generates a test_data.yaml file for YOLOv8 evaluation on the test set.
    """
    yaml_path = os.path.join(save_folder, 'test_data.yaml')
    yaml_content = f"""
path: {save_folder}  # Root directory for dataset
train: images/train  # Placeholder for YOLOv8 requirement
val: images/val      # Placeholder for YOLOv8 requirement
test: detect_images_real
nc: 1
names: ['ball']
"""

    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(yaml_content.strip())
    print(f"test_data.yaml file created at {yaml_path}")

if __name__ == "__main__":
    # Specify the root directory where the dataset is stored
    save_folder = r'C:\Users\20212127\Documents\Y4\BEP\yolov8_training\dataset'
    
    # Create test_data.yaml file
    create_test_yaml_file(save_folder)
