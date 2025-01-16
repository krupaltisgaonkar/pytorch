import argparse
from PIL import Image
import os

def resize_labels_dynamic(input_label_dir, input_image_dir, output_label_dir, new_size):
    """
    Resize YOLO labels dynamically based on the dimensions of the corresponding images.

    Args:
    - input_label_dir (str): Directory containing original YOLO labels.
    - input_image_dir (str): Directory containing original images.
    - output_label_dir (str): Directory to save resized YOLO labels.
    - new_size (tuple): New dimensions for images (width, height).
    """
    os.makedirs(output_label_dir, exist_ok=True)
    new_width, new_height = new_size

    for label_file in os.listdir(input_label_dir):
        if label_file.endswith(".txt"):
            # Get the corresponding image file
            image_file = os.path.splitext(label_file)[0] + ".png"  # Assuming images are .png
            image_path = os.path.join(input_image_dir, image_file)

            # Ensure the image exists
            if not os.path.exists(image_path):
                print(f"Warning: Image file {image_path} not found for label {label_file}. Skipping.")
                continue

            # Get image dimensions
            with Image.open(image_path) as img:
                original_width, original_height = img.size

            # Calculate scaling factors
            scale_x = new_width / original_width
            scale_y = new_height / original_height

            # Process the label file
            input_label_path = os.path.join(input_label_dir, label_file)
            output_label_path = os.path.join(output_label_dir, label_file)

            with open(input_label_path, "r") as infile, open(output_label_path, "w") as outfile:
                for line in infile:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id = parts[0]
                    x_center = float(parts[1]) * original_width
                    y_center = float(parts[2]) * original_height
                    width = float(parts[3]) * original_width
                    height = float(parts[4]) * original_height

                    # Scale values
                    x_center = (x_center * scale_x) / new_width
                    y_center = (y_center * scale_y) / new_height
                    width = (width * scale_x) / new_width
                    height = (height * scale_y) / new_height

                    # Write resized label
                    outfile.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            print(f"Resized labels saved to {output_label_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize YOLO labels based on image dimensions.")
    parser.add_argument("--input_label_dir", type=str, required=True, help="Directory containing original YOLO labels.")
    parser.add_argument("--input_image_dir", type=str, required=True, help="Directory containing original images.")
    parser.add_argument("--output_label_dir", type=str, required=True, help="Directory to save resized YOLO labels.")
    parser.add_argument("--side_length", type=int, required=True, help="New side length because YOLO makes pictures into squares for resizing.")

    args = parser.parse_args()

    resize_labels_dynamic(
        input_label_dir=args.input_label_dir,
        input_image_dir=args.input_image_dir,
        output_label_dir=args.output_label_dir,
        new_size=(args.new_width, args.new_height)
    )
