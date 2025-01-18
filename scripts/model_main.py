import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO  # For YOLOv8
import torch
import time

# Custom models
CUSTOM_MODELS = {
    "custom1": "./models/custom_model1.pt",
    "custom2": "./models/custom_model2.pt",
}

# Predefined YOLO models
YOLO_MODELS = {
    "v5": "ultralytics/yolov5",  # YOLOv5 using PyTorch Hub
    "v8": "ultralytics/yolov8",  # YOLOv8 using Ultralytics
    "v11": "ultralytics/yolov11"  # Hypothetical YOLOv11 (update if available)
}

# Load YOLO model
def load_model(model_name):
    if model_name == "v8":
        print("Loading YOLOv8 model...")
        return YOLO("yolov8n.pt")  # Loads YOLOv8n from Ultralytics
    elif model_name == "v5":
        print("Loading YOLOv5 model...")
        return torch.hub.load("ultralytics/yolov5", "yolov5s")  # YOLOv5 via PyTorch Hub
    elif model_name == "v11":
        print("Loading YOLOv11 model...")
        # Replace 'yolov11n.pt' with the correct path or method to load YOLOv11
        return YOLO("yolo11n.pt")  # Adjust if YOLOv11 uses a different API
    elif model_name in CUSTOM_MODELS:
        custom_path = Path(CUSTOM_MODELS[model_name])
        if custom_path.is_file():
            print(f"Loading custom YOLO model from {custom_path}")
            return YOLO(str(custom_path))  # Load custom YOLO model
        else:
            raise FileNotFoundError(f"Custom model '{model_name}' not found at {custom_path}.")
    else:
        raise ValueError(f"Unrecognized model '{model_name}'. Choose from {list(YOLO_MODELS.keys()) + list(CUSTOM_MODELS.keys())}.")

# Function to process frames with YOLO
def process_frame(model, frame, confidence_threshold=0.5):
    results = model(frame)  # Run YOLO model on the frame
    detections = results[0].boxes  # Access detected boxes (YOLOv8-specific)

    # Filter detections based on confidence threshold
    filtered_boxes = [box for box in detections if box.conf.item() >= confidence_threshold]

    # Draw bounding boxes on the frame
    for box in filtered_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf.item()  # Confidence score
        cls = int(box.cls.item())  # Class ID
        label = f"{model.names[cls]} {conf:.2f}"

        # Draw the box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Process webcam input
def process_webcam(model, webcam_index):
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {webcam_index}.")
        return

    print("Press 'q' to exit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to fetch frame from webcam.")
            break

        # Process frame
        processed_frame = process_frame(model, frame)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(processed_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow(f"Webcam {webcam_index}", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Process image or video files
def process_files(model, input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        cap = cv2.VideoCapture(str(input_path)) if input_path.suffix in [".mp4", ".avi", ".mov"] else None

        if cap:  # Video file
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = process_frame(model, frame)
                cv2.imshow("Video", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
        else:  # Image file
            frame = cv2.imread(str(input_path))
            if frame is not None:
                processed_frame = process_frame(model, frame)
                cv2.imshow("Image", processed_frame)
                cv2.waitKey(0)
    elif input_path.is_dir():  # Process directory
        for file in input_path.iterdir():
            if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                frame = cv2.imread(str(file))
                if frame is not None:
                    processed_frame = process_frame(model, frame)
                    cv2.imshow(f"Image - {file.name}", processed_frame)
                    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--model", required=True, help="Model to use: v5, v8, or custom.")
    parser.add_argument("--input", help="Path to image/video or directory.")
    parser.add_argument("--webcam", type=int, help="Webcam index (0, 1, 2, etc.)")

    args = parser.parse_args()
    model = load_model(args.model)

    if args.webcam is not None:
        process_webcam(model, args.webcam)
    elif args.input:
        process_files(model, args.input)
    else:
        print("Error: Specify --webcam or --input.")
