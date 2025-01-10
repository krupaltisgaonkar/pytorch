# Import necessary libraries
import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
import zipfile

# Define paths
dataset_zip = "dataset.zip"  # Path to the dataset ZIP file
extracted_folder = "dataset"  # Path to extract the dataset
images_folder = os.path.join(extracted_folder, "images")  # Combined images folder
labels_folder = os.path.join(extracted_folder, "labels")  # Combined labels folder
labelmap_file = os.path.join(extracted_folder, "labelmap.pbtxt")  # Path to the labelmap.pbtxt file
classes_file = os.path.join(extracted_folder, "classes.txt")  # Path to the classes.txt file

# Extract dataset.zip
def extract_dataset(zip_path, extract_to):
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    else:
        print(f"Folder {extract_to} already exists. Skipping extraction.")

# Parse labelmap.pbtxt (compatible with any structure)
def parse_labelmap(labelmap_file):
    label_map = {}
    id_to_class = []
    with open(labelmap_file, 'r') as f:
        current_id = None
        current_class = None

        for line in f:
            line = line.strip()
            if "id:" in line:
                current_id = int(line.split(":")[1].strip())
            elif "class:" in line or "name:" in line:
                current_class = line.split(":")[1].strip().replace('"', '').replace("'", "")
                if current_id is not None and current_class is not None:
                    label_map[current_class] = current_id
                    # Ensure the list has enough space for the ID
                    while len(id_to_class) <= current_id:
                        id_to_class.append("")
                    id_to_class[current_id] = current_class
                    current_id = None
                    current_class = None
    return label_map, id_to_class

# Write classes to classes.txt
def write_classes_file(classes_file, id_to_class):
    with open(classes_file, "w") as f:
        for class_name in id_to_class:
            f.write(f"{class_name}\n")

# Convert XML to YOLO format
def convert_xml_to_yolo(xml_file, label_map, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_annotations = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in label_map:
            continue
        class_id = label_map[class_name]

        # Bounding box
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # Convert to YOLO format
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return yolo_annotations

# Process and merge train and test folders
def process_and_merge_folders(base_folder, images_folder, labels_folder, label_map):
    for root_dir, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root_dir, file)
                image_base = os.path.splitext(file)[0]
                img_path = None

                for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                    possible_img_path = os.path.join(root_dir, image_base + ext)
                    if os.path.exists(possible_img_path):
                        img_path = possible_img_path
                        break

                if not img_path:
                    print(f"Image file for {file} not found. Skipping.")
                    continue

                # Copy image to images folder
                os.makedirs(images_folder, exist_ok=True)
                shutil.copy(img_path, os.path.join(images_folder, os.path.basename(img_path)))

                # Get image dimensions
                with Image.open(img_path) as img:
                    img_width, img_height = img.size

                # Convert annotations
                yolo_annotations = convert_xml_to_yolo(xml_path, label_map, img_width, img_height)

                # Save to YOLO .txt file in labels folder
                os.makedirs(labels_folder, exist_ok=True)
                txt_path = os.path.join(labels_folder, image_base + ".txt")
                with open(txt_path, "w") as txt_file:
                    txt_file.write("\n".join(yolo_annotations))

# Main conversion
if __name__ == "__main__":
    # Extract dataset
    extract_dataset(dataset_zip, extracted_folder)

    # Parse label map
    label_map, id_to_class = parse_labelmap(labelmap_file)
    print("Label map:", label_map)

    # Write classes.txt
    write_classes_file(classes_file, id_to_class)
    print(f"Classes written to {classes_file}")

    # Process and merge train and test folders
    print("Processing and merging train and test folders...")
    process_and_merge_folders(extracted_folder, images_folder, labels_folder, label_map)

    print("Conversion and merging complete!")
