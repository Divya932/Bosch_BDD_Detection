import json
import os
from tqdm import tqdm

# Input and output paths
bdd_json_path = '/bosch_od/bosch/data/labels/bdd100k_val.json'
images_dir = '/bosch_od/bosch/data/100k_images/val'
output_coco_path = '/bosch_od/bosch/data/coco_labels/bdd_val_coco.json'

# Define or extract category mapping
CATEGORY_NAMES = set()
with open(bdd_json_path, 'r') as f:
    bdd_data = json.load(f)
    for item in bdd_data:
        for label in item.get("labels", []):
            CATEGORY_NAMES.add(label["category"])

CATEGORY_NAMES = sorted(list(CATEGORY_NAMES))
CATEGORY_ID = {name: idx + 1 for idx, name in enumerate(CATEGORY_NAMES)}  # COCO category IDs start at 1

# Initialize COCO format dict
coco = {
    "images": [],
    "annotations": [],
    "categories": [{"id": v, "name": k} for k, v in CATEGORY_ID.items()]
}

ann_id = 1  # unique annotation ID

for img_id, item in enumerate(tqdm(bdd_data, desc="Converting to COCO")):
    file_name = item["name"]
    image_path = os.path.join(images_dir, file_name)
    if not os.path.exists(image_path):
        continue  # skip missing images

    # Placeholder width/height
    img_width = item.get("width", 1280)
    img_height = item.get("height", 720)

    coco["images"].append({
        "id": img_id,
        "file_name": file_name,
        "width": img_width,
        "height": img_height
    })

    for label in item.get("labels", []):
        if "box2d" not in label:
            continue

        x1 = label["box2d"]["x1"]
        y1 = label["box2d"]["y1"]
        x2 = label["box2d"]["x2"]
        y2 = label["box2d"]["y2"]
        w = x2 - x1
        h = y2 - y1

        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": CATEGORY_ID[label["category"]],
            "bbox": [x1, y1, w, h],
            "area": w * h,
            "iscrowd": 0,
            "segmentation": []
        })
        ann_id += 1

# Save COCO format JSON
with open(output_coco_path, 'w') as f:
    json.dump(coco, f, indent=2)

print(f"COCO annotation saved to: {output_coco_path}")
