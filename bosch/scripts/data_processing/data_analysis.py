import ijson
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def plot_scene_distribution(train_data, val_data, attr="timeofday", save_path=None):
    def get_attr_count(data):
        return Counter(item.get("attributes", {}).get(attr, "unknown") for item in data)

    train_count = get_attr_count(train_data)
    val_count = get_attr_count(val_data)
    categories = sorted(set(train_count) | set(val_count))

    train_vals = [train_count.get(c, 0) for c in categories]
    val_vals = [val_count.get(c, 0) for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, train_vals, width, label="Train", color="skyblue", edgecolor="black")
    plt.bar(x + width / 2, val_vals, width, label="Val", color="orange", edgecolor="black")

    for i, (t, v) in enumerate(zip(train_vals, val_vals)):
        plt.text(i - width / 2, t + 1, str(t), ha='center', fontsize=8)
        plt.text(i + width / 2, v + 1, str(v), ha='center', fontsize=8)

    plt.xticks(x, categories, rotation=45, ha='right')
    plt.title(f"{attr.capitalize()} Distribution in Train/Val")
    plt.xlabel(attr.capitalize())
    plt.ylabel("Image Count")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved {attr} distribution plot to {save_path}")
    else:
        plt.show()


def plot_bbox_size_distribution(train_data, val_data, bins=30, normalize=True, save_path=None):
    def extract_bbox_areas(data):
        areas = []
        for item in data:
            img_width = item.get("width", 1)
            img_height = item.get("height", 1)
            for label in item.get("labels", []):
                box2d = label.get("box2d", {})
                if all(k in box2d for k in ["x1", "y1", "x2", "y2"]):
                    w = max(box2d["x2"] - box2d["x1"], 0)
                    h = max(box2d["y2"] - box2d["y1"], 0)
                    area = w * h
                    if normalize:
                        area /= (img_width * img_height)
                    areas.append(area)
        return areas

    #train_data, val_data = read_json(train_json), read_json(val_json)
    train_areas = extract_bbox_areas(train_data)
    val_areas = extract_bbox_areas(val_data)

    plt.figure(figsize=(10, 6))
    plt.hist(train_areas, bins=bins, alpha=0.6, label="Train", color="skyblue", edgecolor="black")
    plt.hist(val_areas, bins=bins, alpha=0.6, label="Val", color="orange", edgecolor="black")
    plt.xlabel("Normalized Bounding Box Area" if normalize else "Bounding Box Area (px²)")
    plt.ylabel("Frequency")
    plt.title("Bounding Box Size Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved bbox size distribution plot to {save_path}")
    else:
        plt.show()


def plot_class_distribution(class_count, title="BDD Class Distribution", save_path = None):
    plt.figure(figsize=(14, 6))
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')

    if len(class_count) == 2:
        #assuming idx 0 has train and idx 1 has val
        classes = sorted(set(class_count[0].keys()).union(set(class_count[1].keys())))

        train_counts = list(class_count[0].values())
        val_counts = list(class_count[1].values())

        x = np.arange(len(classes))  # label locations
        width = 0.35  # width of bars

        bars1 = plt.bar(x - width/2, train_counts, width, label='Train', color='skyblue', edgecolor='black')
        bars2 = plt.bar(x + width/2, val_counts, width, label='Val', color='orange', edgecolor='black')

        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            train = train_counts[i]
            val = val_counts[i]
            total = train + val

            # Avoid division by zero
            if total > 0:
                train_pct = round(100 * train / total)
                val_pct = 100 - train_pct  # ensures sum to 100
            else:
                train_pct = val_pct = 0

            # Add count + percent above each bar
            if train > 0:
                plt.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height() - 10,
                        f"({train_pct}%)", ha='center', va='bottom', fontsize=8)
            if val > 0:
                plt.text(bar2.get_x() + bar2.get_width() / 2, bar2.get_height() - 10,
                        f"({val_pct}%)", ha='center', va='bottom', fontsize=8)
            
        plt.xticks(ticks=x, labels= classes, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

    else:
        classes = list(class_count.keys())
        counts = list(class_count.values())

        bars = plt.bar(classes, counts, color='skyblue', edgecolor='black')

        # Add value labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, int(yval), ha='center', va='bottom')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()


    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def read_json(json_path):
    with open(json_path, 'r') as jf:
        contents = jf.read()
        if not contents.strip():
            print("ERROR: File is empty!")
            return
        data = json.loads(contents)
    
    return data

def count_classes(json_path):
    json_dict = read_json(json_path)
    class_count = {}
    for idx, item_dict in enumerate(json_dict):
        for label in item_dict.get("labels", []):
            category = label.get("category")
            if category:
                if category in class_count.keys():
                    class_count[category] += 1
                else:
                    class_count[category] = 1

    print("Final classes: ", class_count.keys(), len(class_count))
    print("Total entries: ", idx)
    print(class_count)

    return class_count


def main():
    train_json_path = "/bosch_od/bosch/data/labels/bdd100k_train.json"
    val_json_path = "/bosch_od/bosch/data/labels/bdd100k_val.json"

    train_data, val_data = read_json(train_json_path), read_json(val_json_path)

    train_class_count = count_classes(train_json_path)
    val_class_count = count_classes(val_json_path)

    plot_class_distribution(class_count = train_class_count, save_path = "/bosch_od/bosch/outputs/class_dist_train.png")
    plot_class_distribution(class_count = val_class_count, save_path = "/bosch_od/bosch/outputs/class_dist_val.png")
    plot_class_distribution(class_count = [train_class_count, val_class_count], save_path = "/bosch_od/bosch/outputs/class_dist_train_val.png")
    plot_bbox_size_distribution(train_data, val_data, bins=30, normalize=True, save_path="/bosch_od/bosch/outputs/bbox_size_distribution.png")
    plot_scene_distribution(train_data, val_data, attr="scene", save_path="/bosch_od/bosch/outputs/scene_distribution.png")

if __name__ == "__main__":
    main()
