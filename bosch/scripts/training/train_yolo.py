import os
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import cv2

# --- Config ---
MODEL_PATH = "/bosch_od/bosch/model/yolov11m.pt"  # or your trained model: "runs/detect/train/weights/best.pt"
IMAGE_DIR = "/bosch_od/bosch/data/100k_images/val"  # Path to BDD val images
OUTPUT_DIR = "/bosch_od/bosch/outputs/pictures"   # Where to save results
CONF_THRESHOLD = 0.25  # Confidence threshold

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# List of image files
image_paths = sorted(list(Path(IMAGE_DIR).glob("*.jpg")))

# Run inference
for img_path in tqdm(image_paths, desc="Running inference"):
    results = model(img_path, conf=CONF_THRESHOLD)

    # Save results (optional)
    for r in results:
        # Render predictions on image
        rendered = r.plot()
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
        cv2.imwrite(out_path, rendered)

print(f"Inference completed. Results saved to: {OUTPUT_DIR}")
