# data_analysis.py
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict


def read_json(json_path):
    with open(json_path, 'r') as jf:
        contents = jf.read()
        if not contents.strip():
            raise ValueError(f"File {json_path} is empty!")
        return json.loads(contents)


def count_classes(data):
    """
    Count number of objects per class in a list of annotations.
    """
    class_counts = defaultdict(int)
    for item in data:
        for label in item.get('labels', []):
            class_name = label.get('category', 'unknown')
            class_counts[class_name] += 1
    return dict(class_counts)


def plot_class_distribution(class_count, title="Class Distribution"):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(title)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')

    if isinstance(class_count, list) and len(class_count) == 2:
        classes = sorted(set(class_count[0].keys()).union(set(class_count[1].keys())))
        train_counts = [class_count[0].get(cls, 0) for cls in classes]
        val_counts = [class_count[1].get(cls, 0) for cls in classes]

        x = np.arange(len(classes))
        width = 0.35

        ax.bar(x - width / 2, train_counts, width, label='Train', color='skyblue')
        ax.bar(x + width / 2, val_counts, width, label='Val', color='orange')

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
    else:
        classes = list(class_count.keys())
        counts = list(class_count.values())
        ax.bar(classes, counts, color='skyblue')
        ax.set_xticklabels(classes, rotation=45, ha='right')

    plt.tight_layout()
    return fig


def plot_bbox_size_distribution(data, bins=30, normalize=True):
    areas = []
    for item in data:
        img_w = item.get("width", 1)
        img_h = item.get("height", 1)
        for label in item.get("labels", []):
            box2d = label.get("box2d", {})
            if all(k in box2d for k in ["x1", "y1", "x2", "y2"]):
                w = max(box2d["x2"] - box2d["x1"], 0)
                h = max(box2d["y2"] - box2d["y1"], 0)
                area = w * h
                if normalize:
                    area /= (img_w * img_h)
                areas.append(area)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(areas, bins=bins, color='skyblue', edgecolor='black')
    ax.set_xlabel("Normalized Bounding Box Area" if normalize else "Bounding Box Area (px²)")
    ax.set_ylabel("Frequency")
    ax.set_title("Bounding Box Size Distribution")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_scene_distribution(data, attr="scene"):
    def get_attr_count(data):
        return Counter(item.get("attributes", {}).get(attr, "unknown") for item in data)

    count = get_attr_count(data)
    categories = sorted(count.keys())
    values = [count[c] for c in categories]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(categories, values, color='skyblue')
    ax.set_title(f"{attr.capitalize()} Distribution")
    ax.set_xlabel(attr.capitalize())
    ax.set_ylabel("Image Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig
