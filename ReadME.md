## BDD-Object-Detection: Assignment
This assignment aims to study the BDD 100k Dataset for Object Detection classes and run a round of training over the dataset.  

# BDD100K Dataset Analysis
This work provides an in-depth analysis of the **BDD100K** dataset, focusing on various factors available in the data annotations like class, scene, time of day, and weather distributions. The goal is to understand the data distribution and characteristics that may influence training and model generalization.

---

## 📊 1. Class Distribution
This dataset contains a total of 70,000 train images and 10,000 val images. The distribution is across 12 classes - bike, bus, car, drivable_area, lane, motor, person, rider, traffic_light, traffic_sign, train, truck. Out of these two classes belong to semantic segmentation - drivable_area and lane. So we consider the remaining 10 classes for our performance analysis. 

### Combined Train and Validation Sets
![class_dist_train_val.png](./assets/train_val_analysis.png)

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
![Scene_distribution.png](./Scene_distribution.png)

- **Most Common**: `city street` scenes dominate the dataset.
- **Rare Scenes**: `tunnel`, `parking lot`, and `gas stations` occur infrequently.

- **Implication**:  
  Models may overfit to urban driving scenarios. Specialized tuning or few-shot learning methods might help improve rare scene recognition.

---

## 🌅 3. Time of Day Distribution
![Time_distribution.png](./Time_distribution.png)

- **Distribution**:
  - `daytime`: Majority of images.
  - `night`: Moderately represented.
  - `dawn/dusk` and `undefined`: Rare.

- **Impact**:
  - Models trained primarily on daytime images may struggle in night or dawn/dusk conditions.
  - We could consider using image enhancement or nighttime-specific models if deployment involves varied lighting. 

---

## 🌦️ 4. Weather Distribution
![weather_distribution.png](./weather_distribution.png)

- **Dominated by**: `clear` weather.
- **Underrepresented**: `foggy`, `snowy`, and `rainy` conditions.

- **Implication**:  
  The dataset may not adequately prepare models for real-world deployment under poor weather conditions.

- **Remedies**:
  - Include more diverse weather conditions using simulation
  - Weather augmentation (e.g., fog and rain overlays).
  - Domain adaptation from synthetic-to-real weather.

---

## ✅ Summary & Recommendations

- Significant **class and condition imbalance** exists in BDD100K.
- Bias toward **urban, clear, daytime conditions** may limit model generalization.
- Performance on **rare classes and conditions** may be low without countermeasures.

### 📌 Recommendations

- Apply **class-aware loss functions** and sampling strategies.
- Use **augmentation and synthetic data** to improve representation of rare scenarios.
- Evaluate models with **per-class and per-condition metrics**, not just overall accuracy.
- Consider building a **balanced validation subset** for fair evaluation.

---

