## BDD-Object-Detection: Assignment
This assignment aims to study the BDD 100k Dataset for Object Detection classes and run a round of training over the dataset.  

# BDD100K Dataset Analysis
This work provides an in-depth analysis of the **BDD100K** dataset, focusing on various factors available in the data annotations like class, scene, time of day, and weather distributions. The goal is to understand the data distribution and characteristics that may influence training and model generalization.

---

## 📊 1. Class Distribution
This dataset contains a total of 70,000 train images and 10,000 val images. The distribution is across 12 classes - bike, bus, car, drivable_area, lane, motor, person, rider, traffic_light, traffic_sign, train, truck. Out of these two classes belong to semantic segmentation - drivable_area and lane. So we consider the remaining 10 classes for our performance analysis. 

### Combined Train and Validation Sets
![class_dist_train_val.png](./assets/train_val_hist.png)

- **Dominant Classes**:  
  `car`, `lane`, and `drivable area` appear most frequently in the dataset, with `car` appearing in over 70% of samples.
  
- **Underrepresented Classes**:  
  `train`, `motor`, `bike`, and `rider` are severely underrepresented, often with less than 1% presence.

- **Observation**:  
  The extreme skew towards common classes indicates a **long-tail distribution**, which can cause significant performance drop for rare class predictions.

- **Train/Val split**:
  If we observe the train/val split of the current set, `person` class is totally under-represented in the train set and significantly present in the val set.
  This can lead to wrong validation over the concerned class as there is lacking training data, but significant presence in validation set.

  The train/val split across classes is uneven.  
  1. bike -  84:16
  2. bus - 90:10
  3. car - 87:13
  4. motor - 84:16
  5. person - 13:87
  6. rider - 69:31
  7. traffic_light - 74:26
  8. traffic_sign - 63:37
  9. train - 75:25
  10. truck - 90:10
 
  The above distribution should have similar numbers across all classes, ideally a split of 70:30 or 80:20.  

---

## 🧠 Class Imbalance: Detailed Analysis

- **Challenges Imposed**:  
  - Models tend to bias predictions toward frequent classes.  
  - Rare classes may have high false negative rates.  
  - Evaluation metrics (like mAP in object detection case) can be misleading if dominated by frequent classes.  

- **Mitigation Strategies**:  
  - **Class-aware sampling**: Oversample rare classes or undersample dominant ones during training.  
  - **Loss weighting**: Use inverse frequency or log-frequency class weights in the loss function (e.g., in CrossEntropy or Focal Loss).  
  - **Focal Loss**: Helps the model focus more on hard, misclassified examples.  
  - **Data augmentation**: Apply augmentations only to rare classes (copy-paste, GANs, etc.).  
  - **Synthetic data generation**: Create synthetic examples of rare classes using domain randomization or simulation.  
  - **Two-stage detectors**: First stage detects common objects; second stage focuses on hard or rare classes.  

---

## 🏙️ 2. Scene Type Distribution
![Scene_distribution.png](./assets/scene_dist.png)

- **Most Common**: `city street` scenes dominate the dataset.  
- **Rare Scenes**: `tunnel`, `parking lot`, and `gas stations` occur infrequently.   

- **Implication**:  
  Models may overfit to urban driving scenarios. Specialized tuning or few-shot learning methods might help improve rare scene recognition.    

---

## 🌅 3. Time of Day Distribution
![Time_distribution.png](./assets/time_dist.png)  

- **Distribution**:
  - `daytime`: Majority of images.    
  - `night`: Moderately represented.  
  - `dawn/dusk` and `undefined`: Rare.  

- **Impact**:
  - Models trained primarily on daytime images may struggle in night or dawn/dusk conditions.  
  - We could consider using image enhancement or nighttime-specific models if deployment involves varied lighting.    

---

## 🌦️ 4. Weather Distribution
![weather_distribution.png](./assets/weather_dist.png)

- **Dominated by**: `clear` weather.
- **Underrepresented**: `foggy`, `snowy`, and `rainy` conditions.  

- **Implication**:  
  The dataset may not adequately prepare models for real-world deployment under poor weather conditions.  

- **Remedies**:
  - Include more diverse weather conditions using simulation   
  - Weather augmentation (e.g., fog and rain overlays).   
  - Domain adaptation from synthetic-to-real weather.  
  
---

### Data Analysis code   
#### Docker Setup
Please clone the repository on your system and build the docker image using the Dockerfile provided with this repository.    
The Dockerfile provides environment for Data Analysis, Model training and evaluation, and dashboarding.    
`docker build -t bdd100k_image_divya .`   
`docker run -it -p 8501 -v $(pwd):/bosch_od --gpus all --name bdd100k_divya_container bdd100k_image_divya /bin/bash`   

Make sure to place train and val labels json files at `/bosch_od/bosch/data/labels/bdd100k_train.json` and `/bosch_od/bosch/data/labels/bdd100k_val.json`   

- Run `scripts/data_processing/data_analysis.py` to save analysis results to disk.       
- Run `cd scripts/data_processing && streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501` to visualize the data on a dashboard.   

Here are some screenshots of the Dashboard as it runs on your localhost   
![Dashboard1.png](./assets/streamlit_app.png) ![Dashboard.png](./assets/streamlit_app2.png)   


## Model Selection - Yolov8m/Yolov11m - Reasoning    
### 🧠 Why YOLOv8 for Object Detection?  

For this BDD100K object detection project, we selected **YOLOv8m (You Only Look Once version 8)** as the model of choice. It balances between **speed**, **accuracy**, **deployment readiness**, and **ease of use**. YOLOv8 also introduces architectural improvements over previous YOLO versions, making it ideal for large-scale traffic scene understanding tasks.   

This project also experiments with **Yolov11** for a comparison over the BDD100k validation set. Although the differences observed in performance are not huge, but it could be immensely different with an improved data distribution as noted in above sections.

---

### ✅ Key Advantages of YOLOv8m/YOLOv11m  

- **Real-time inference** (30–60 FPS on modern GPUs, 15+ FPS on edge devices)  
- **High accuracy** on small and occluded objects (common in BDD100K)  
- **Modular architecture** with support for detection, segmentation, classification  
- **Built-in tools**: AutoAnchor tuning, augmentation, visualization, logging  
- **Easy deployment**: Export to ONNX, TensorRT, TFLite, CoreML  
- **Large community** and Ultralytics support for quick debugging and updates   

---

### Architecture Comparison: YOLOv8/YOLOv11m vs DETR / Transformer Detectors  

| Feature | **YOLOv8** | **DETR / DINO / Transformer-based** |  
|--------|------------|--------------------------------------|  
| **Backbone** | CSPDarknet, C2f blocks, lightweight ConvNets | ResNet-50, Swin, ViT, or other transformers |    
| **Neck** | PAN-FPN or BiFPN-like structure for feature fusion | No neck (or simple projection heads) |   
| **Head** | Decoupled classification and regression heads | Transformer decoder layers + FFN |   
| **Detection strategy** | Anchor-free dense prediction with NMS | Set-based matching with Hungarian loss |   
| **Positional encoding** | Implicit via spatial conv layers | Explicit 2D positional encodings |   
| **Training time** | Fast (~6–12 hrs for large datasets) | Long (~24–72 hrs or more) |   
| **Data requirements** | Efficient even with medium data | Requires large datasets and epochs to converge |   
| **Inference speed** | Real-time, optimized | Slow (~3–10 FPS on GPU) |   
| **Object count limitation** | Dynamic with NMS | Fixed-size (e.g., 100 predictions per image) |   
| **Output stability** | High consistency | Output can fluctuate with small input shifts |   
| **Deployment** | Easy (ONNX, TensorRT, CoreML) | Complex due to attention layers, dependencies |   

---

While transformer-based models like **DETR** and **DINO** are excellent for high-resource setups, yolov8/yolov11 has much **faster inference speed**, **simple training pipeline**, and **easy deployment setup** which makes it **more suitable** for large-scale real-time applications like BDD100K driving datasets. 

---

### ✅ Final Decision: YOLOv8/Yolov11

Given the BDD100K dataset's:
- **Large image variety and class imbalance**
- **Need for real-time predictions for deployment on edge devices**
- **Multiple small, occluded, and overlapping objects**

---   


## Model Training - Yolov11  
Setting up training for Yolov8/Yolov11 includes two major steps:  
- preparing data and annotations from BDD to Yolo format - run script at `bosch/scripts/data_processing/convert_to_yolo.py`
- Run training script at `bosch/scripts/data_processing/train.py`

Here's a screenshot of the training initiated after data preparation for this project - 
![Training.png](./assets/training_on_2GPUs.png)  

This training was initiated on 2GPUs - availing DDP - Data Distributed Parallel training from Ultralytics pipeline.

## ✅ Evaluation and Visualization   

### 1. **Precision-Recall Curve (PR Curve)** 
![pr_curve.png](./assets/pr_curve.png)
- **Best performing classes**:  
  - `car`: mAP@0.5 ≈ **0.636**  
  - `truck`: ≈ **0.398**  
  - `motorcycle`: ≈ **0.355**
- **Weak classes**:  
  - `bicycle`, `airplane`, `bus`, `boat` have **0.0 mAP**
  - `person`, `train`, `traffic light` show **poor precision and recall**
- **Overall mAP@0.5**: **0.143** — indicates **low detection quality across most classes**

### 2. **Recall-Confidence Curve**  
![rc_curve.png](./assets/rc_curve.png)
- High recall at **low confidence** thresholds for most classes, but rapidly drops with increasing confidence.
- Model outputs many detections with **low confidence**, leading to potential false positives.

### 3. **Precision-Confidence Curve** 
![pc_curve.png](./assets/pc_curve.png)
- **Precision is low** at low confidence, only improves at high thresholds (~0.9+).
- Indicates that many predictions are incorrect unless filtered by very high confidence.

### 4. **F1-Confidence Curve** 
![f1_curve.png](./assets/f1_curve.png)
- **Peak F1-score** ≈ **0.16 at 0.235 confidence**
- Low F1 indicates poor balance between precision and recall.
- Suggests optimal confidence threshold may be around **0.2–0.3**

---

## 🏆 Best Metric to Use: `mAP@0.5`
- It offers a **balanced measure** of precision and recall across all classes.
- Aggregates performance over multiple thresholds.
- Identifies **per-class weaknesses** and overall detection quality.
- It is a **standard metric** in object detection benchmarks like COCO and BDD.

While F1-score and confidence-based curves help in **threshold tuning**, **`mAP@0.5` gives the best high-level insight** into model performance.

---  

### Comparison between YOLOv8 and YOLOv11  
This project compares both YOLOv8m and YOLOv11m for the choice of model. However the performance appears to be quite similar.  
![yolov8_val.png](./assets/yolov8_val.png)  ![yolov11_val.png](./assets/yolov11_val.png)  

YOLOv11 builds upon YOLOv8, hence it definitely offers improvements in speed, efficiency, and accuracy. It would be a better choice for these reasons during training. 
## Possible Improvements

### 📦 Data Quality & Distribution
- **Class imbalance**:
  - Oversample underrepresented classes like `bus`, `boat`, `airplane`
  - Use class-specific augmentation to improve data diversity
- **Annotation quality**:
  - Review dataset for label errors or missing annotations
- **Object size/visibility issues**:
  - Small or occluded objects may require better resolution handling or image cropping

### 🧠 Model Enhancements
- **Anchor tuning**: Adjust anchors to better match BDD object sizes
- **Data augmentation**:
  - Use techniques like mosaic, cutmix, mixup, and color jitter
- **Weighted Loss functions**:
  - Increase class weights for rare classes to counter imbalance
- **Model ensemble**:
  - Use multiple models or checkpoints to boost per-class accuracy

---


## ✅ Summary & Possible Improvements  
The YOLOv8 model shows strong performance on frequent classes like `car` and `truck`, but struggles on rare and small-object classes. The overall mAP@0.5 = **0.143** suggests a need for:
- **Balanced data curation**
- **Improved training strategies**
- **Model tuning and augmentation techniques**

- Significant **class and condition imbalance** exists in BDD100K.    
- Bias toward **urban, clear, daytime conditions** may limit model generalization.    
- Performance on **rare classes and conditions** may be low without countermeasures.      

### Improvements

- Apply **class-aware loss functions** and sampling strategies.  
- Use **augmentation and synthetic data** to improve representation of rare scenarios.   
- Evaluate models with **per-class and per-condition metrics**, not just overall accuracy.   
- Building a **balanced validation subset** for fair evaluation.
- **Longer training** (more epochs) to improve convergence
- **Better LR scheduling** (e.g., cosine annealing, warmup)
- **Finetune pretrained weights** for better feature generalization on BDD

--- 

