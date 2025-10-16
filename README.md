# 🩻 Chest X-Ray Pneumonia Detection

This project uses a **Convolutional Neural Network (CNN)** to classify chest X-ray images as **Normal** or **Pneumonia**.

---

## 📂 Dataset
The dataset is publicly available on **Kaggle**:

👉 [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

It contains:
- **Train set:** X-ray images for model training  
- **Test set:** Images for final evaluation  
- **Validation set:** For tuning the model  

---

## 🧠 Model Architecture
We start with a custom CNN, then improve performance using **MobileNetV2** (transfer learning).

**Model Layers:**
- Conv2D → MaxPooling  
- Conv2D → MaxPooling  
- Flatten → Dense → Dropout  
- Output (Sigmoid activation)

---

## ⚙️ Preprocessing Steps
- Resize images to **224×224**
- Normalize pixel values (0–1)
- Apply data augmentation (rotation, zoom, flip)

---

## 🎯 Evaluation
We used:
- Accuracy, Precision, Recall, F1-score  
- Confusion Matrix visualization  
- Training vs Validation curves  

---

## 🚀 Results
| Metric | Score |
|---------|--------|
| Accuracy | 0.50 (Initial CNN) |
| Improved (MobileNetV2) | *Expected 90%+* |

---

## 🧩 Tools Used
- Python 3  
- TensorFlow / Keras  
- NumPy, Matplotlib, Seaborn  
- Google Colab  
- Kaggle Dataset  

---

## 👨‍💻 Author
**Mhmd Safa**  
Deep Learning Research | Image Classification | Model Deployment
