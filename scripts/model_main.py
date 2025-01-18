"""
Use a YOLO Predefined Model (v5, v8) with Webcam
python yolo_input_script.py --model v8 --webcam 0

Use a YOLO Predefined Model with an Image or Video
python yolo_input_script.py --model v5 --input ./data/image.jpg
python yolo_input_script.py --model v8 --input ./data/video.mp4

Use a Custom YOLO Model
python yolo_input_script.py --model custom1 --input ./data/image.jpg

CHANGE THE CUSTOM MODELS VARIABLE TO YOUR ACTUAL MODELS!!!!!!!!
"""

import argparse
import cv2
import os
from pathlib import Path
from ultralytics import YOLO  # Use for YOLOv8
import torch

# Predefined YOLO models
YOLO_MODELS = {
    "v5": "ultralytics/yolov5",  # YOLOv5 using PyTorch Hub
    "v8": "ultralytics/yolov8",# YOLOv8 using Ultralytics
    "v11": "ultralytics/yolov11"
}

# Custom models directory
CUSTOM_MODELS = {
    "custom1": "./models/custom_model1.pt",
    "custom2": "./models/custom_model2.pt"
}

# Function to load YOLO model
def load_model(model_name):
    if model_name == "v8":
        print(f"Loading YOLOv8 model using Ultralytics...")
        return YOLO("yolov8n.pt")  # Loads YOLOv8n from Ultralytics
    elif model_name == "v5":
        print(f"Loading YOLOv5 model using PyTorch Hub...")
        return torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 using PyTorch Hub
    elif model_name == "v11"
        print(f"Lodaing YOLOv11 model using Ultralytics")
        return YOLO("yolov11n.pt")
    elif model_name in CUSTOM_MODELS:
        custom_path = Path(CUSTOM_MODELS[model_name])
        if custom_path.is_file():
            print(f"Loading custom YOLO model: {model_name} from {custom_path}")
            return YOLO(str(custom_path))  # Load custom YOLO model
        else:
            raise FileNotFoundError(f"Custom model '{model_name}' not found at {custom_path}.")
    else:
        raise ValueError(f"Model '{model_name}' is not recognized. Available options are: {list(YOLO_MODELS.keys()) + list(CUSTOM_MODELS.keys())}.")

# Function to process frames with YOLO
def process_frame(model, frame):
    # Inference using the YOLO model
    results = model(frame)  # Perform detection
    if isinstance(results, list):  # YOLOv5 case
        return results.render()[0]  # Render the frame
    else:  # YOLOv8 case
        return results[0].plot()  # Render the frame with bounding boxes

# Process webcam input
def process_webcam(model, webcam_index):
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {webcam_index}.")
        return

    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to fetch frame from webcam.")
            break

        processed_frame = process_frame(model, frame)
        cv2.imshow(f"Webcam {webcam_index}", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Process video or image files
def process_files(model, input_path):
    input_path = Path(input_path)

    if input_path.is_file():
        cap = cv2.VideoCapture(str(input_path)) if input_path.suffix in ['.mp4', '.avi', '.mov'] else None

        if cap:  # Video file
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = process_frame(model, frame)
                cv2.imshow("Video", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
        else:  # Image file
            frame = cv2.imread(str(input_path))
            if frame is not None:
                processed_frame = process_frame(model, frame)
                cv2.imshow("Image", processed_frame)
                cv2.waitKey(0)

    elif input_path.is_dir():
        for file in input_path.iterdir():
            if file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                frame = cv2.imread(str(file))
                if frame is not None:
                    processed_frame = process_frame(model, frame)
                    cv2.imshow(f"Image - {file.name}", processed_frame)
                    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Object Detection Script")
    parser.add_argument("--model", required=True, help="YOLO model to use: predefined model name (v5, v8) or custom model name from predefined list.")
    parser.add_argument("--input", help="Path to an image, video, or directory containing images.")
    parser.add_argument("--webcam", type=int, help="Webcam index (0, 1, 2, etc.).")

    args = parser.parse_args()

    # Load YOLO model
    model = load_model(args.model)

    if args.webcam is not None:
        process_webcam(model, args.webcam)
    elif args.input:
        process_files(model, args.input)
    else:
        print("Error: You must specify either --webcam or --input.")
