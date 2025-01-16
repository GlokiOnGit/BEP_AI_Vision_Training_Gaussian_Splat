import os
import cv2
from ultralytics import YOLO

def main():
    # Load the trained YOLO model
    model = YOLO(r'C:\Users\20212127\Documents\Y4\BEP\Training\thursday_28_11_2024_1.pt')

    # Define paths for input video and output
    video_path = r'C:\Users\20212127\Documents\Y4\BEP\practice_match\dynamic_match\play_1.mp4'
    output_video_path = r'C:\Users\20212127\Documents\Y4\BEP\yolov8_training\dataset\output_video_detection.mp4'

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

    # Initialize video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Perform detection
        results = model(frame, conf=0.9)

        # Draw bounding boxes on the frame
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video
        out.write(annotated_frame)

    # Release resources
    cap.release()
    out.release()

    print(f"Annotated video saved to {output_video_path}")

if __name__ == '__main__':
    main()
