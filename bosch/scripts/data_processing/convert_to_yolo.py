import os
import json
from tqdm import tqdm

# Class name to YOLO class ID mapping
class_map = {
    'bike': 0,
    'bus': 1,
    'car': 2,
    'motor': 3,
    'person': 4,
    'rider': 5,
    'train': 6,
    'truck': 7,
    'traffic light': 8,
    'traffic sign': 9
}

# Paths
json_path = '/bosch_od/bosch/datasets/labels/bdd100k_train.json'  # BDD100K format JSON
images_dir = '/bosch_od/bosch/datasets/100k_images/train'  # directory with images
labels_dir = '/bosch_od/bosch/datasets/100k_images/train/labels'  # output labels

os.makedirs(labels_dir, exist_ok=True)

# Load JSON
with open(json_path, 'r') as f:
    annotations = json.load(f)

# Convert each entry
for item in tqdm(annotations, desc="Converting"):
    image_name = item['name']
    image_w, image_h = 1280, 720  # BDD100K standard resolution

    label_lines = []
    for label in item.get('labels', []):
        category = label['category']
        if category not in class_map:
            continue

        cls_id = class_map[category]
        box = label.get('box2d')
        if not box:
            continue

        # Extract and normalize bbox
        x1, y1 = box['x1'], box['y1']
        x2, y2 = box['x2'], box['y2']
        cx = (x1 + x2) / 2 / image_w
        cy = (y1 + y2) / 2 / image_h
        w = (x2 - x1) / image_w
        h = (y2 - y1) / image_h

        label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    # Write .txt file
    base = os.path.splitext(image_name)[0]
    label_path = os.path.join(labels_dir, base + '.txt')
    with open(label_path, 'w') as f:
        f.write('\n'.join(label_lines))
