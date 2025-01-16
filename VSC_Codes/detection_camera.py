import cv2
from ultralytics import YOLO

def main():
    # Load the trained YOLO model
    model = YOLO(r'C:\Users\20212127\Documents\Y4\BEP\Training\trainings\orange_ball_background.pt')

    # Initialize webcam (use 0 for the default camera, or specify the index for an external camera)
    cap = cv2.VideoCapture(1)  # Change the argument to the camera index if necessary

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    # Define the codec and create VideoWriter object
    output_path = r'C:\Users\20212127\Documents\Y4\BEP\yolov8_training\dataset\output_detection.avi'  # Change the filename as needed
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
    fps = 20.0  # Frames per second
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Press 'q' to exit the live feed.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from the camera.")
            break

        # Perform detection on the current frame
        results = model(frame, conf=0.6)

        # Annotate the frame with detection results
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video file
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam, writer, and close all OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {output_path}")

if __name__ == '__main__':
    main()
