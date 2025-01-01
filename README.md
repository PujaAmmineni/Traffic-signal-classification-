# Traffic Signal Classification

This repository implements a machine learning-based system for **Traffic Signal Classification** to enhance the functionality of autonomous vehicles. The project leverages image preprocessing, feature extraction, and supervised learning to classify traffic signs accurately.

## **Abstract**
In the era of autonomous vehicles, the accurate classification of traffic signals is critical for road safety and efficient traffic management. This system uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset and explores several machine learning classifiers to achieve high accuracy in classifying traffic signs.

---

## **Key Features**
- **Image Preprocessing**: Normalization, resizing (32x32x3), and conversion to PNG format.
- **Feature Extraction**: Utilizes the **Img2Vec tool** to generate 512-dimensional feature vectors from traffic sign images.
- **Machine Learning Models**:
  - Decision Tree
  - Random Forest
  - Gaussian Naïve Bayes
  - Multi-Layer Perceptron (MLP)
  - Support Vector Classifier (SVC)
- **Evaluation Metrics**: Accuracy, Precision, Recall, and F1 Score.

---

## **Dataset**
- **Source**: [German Traffic Sign Recognition Benchmark (GTSRB)](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html)
- **Description**: 
  - ~34,000 images across 42 traffic sign classes.
  - Includes varying environmental and lighting conditions.
- **Preprocessing Steps**:
  - Resized to `32x32x3` (RGB format).
  - Converted to PNG format.

---

## **Methodology**
### **Workflow Overview**
1. **Preprocessing**:
   - Normalize and resize images.
   - Convert images to feature vectors using Img2Vec.
2. **Model Training**:
   - Train various classifiers with stratified k-fold cross-validation.
   - Standardize features using Standard Scaler.
3. **Model Evaluation**:
   - Evaluate models on metrics such as accuracy, precision, recall, and F1 Score.
   - Address class imbalance using stratified validation.

---

## **Models and Results**
### **Performance Metrics**
| Model                   | Accuracy | Precision | Recall | F1 Score |
|--------------------------|----------|-----------|--------|----------|
| Decision Tree            | 59%      | 53%       | 53%    | 53%      |
| Random Forest            | 91%      | 95%       | 86%    | 89%      |
| Gaussian Naïve Bayes     | 71%      | 71%       | 75%    | 73%      |
| Multi-Layer Perceptron   | 96%      | 97%       | 96%    | 97%      |
| Support Vector Classifier| **98%**  | **99%**   | **98%**| **98%**  |

### **Best Model**
- **Support Vector Classifier (SVC)**:
  - Highest performance with 98% accuracy.
  - Robust in handling high-dimensional, imbalanced datasets.

---

## **Installation and Usage**
### **Prerequisites**
- Python 3.8+
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `img2vec`.

