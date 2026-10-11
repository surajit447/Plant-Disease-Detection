# 🌿 Plant Disease Detection

A deep learning project that identifies **38 plant leaf conditions** (diseases and healthy leaves) across **14 crops** from a single photo. It uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras and reaches about **95% accuracy on the validation set**.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Supported Crops and Diseases](#-supported-crops-and-diseases)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Making a Prediction](#-making-a-prediction)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

Plant diseases reduce crop yield and quality, and spotting them early is key to protecting harvests. This project trains an image classifier that takes a photo of a leaf and predicts which disease (or healthy state) it shows.

The workflow has two notebooks:

1. **`Train_plant_disease.ipynb`** loads the dataset, builds and trains the CNN, evaluates it (accuracy, classification report, confusion matrix) and saves the trained model.
2. **`Test_Plant_Disease.ipynb`** loads the saved model and predicts the disease for a single leaf image.

---

## 📂 Dataset

The model is trained on the **New Plant Diseases Dataset** (an augmented version of the PlantVillage dataset) from Kaggle:

🔗 https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

| Split        | Images  | Classes |
|--------------|---------|---------|
| Training     | 70,295  | 38      |
| Validation   | 17,572  | 38      |
| Test samples | 33      | n/a (unlabelled sample images) |

> **Note:** The dataset is large and is **not included in this repository**. Download it from Kaggle and place it as described in [Getting Started](#-getting-started).

---

## 🌱 Supported Crops and Diseases

| Crop | Conditions |
|------|-----------|
| Apple | Apple scab, Black rot, Cedar apple rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery mildew, Healthy |
| Corn (maize) | Cercospora leaf spot / Gray leaf spot, Common rust, Northern leaf blight, Healthy |
| Grape | Black rot, Esca (Black measles), Leaf blight (Isariopsis leaf spot), Healthy |
| Orange | Haunglongbing (Citrus greening) |
| Peach | Bacterial spot, Healthy |
| Pepper (bell) | Bacterial spot, Healthy |
| Potato | Early blight, Late blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery mildew |
| Strawberry | Leaf scorch, Healthy |
| Tomato | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites (two-spotted), Target spot, Yellow leaf curl virus, Mosaic virus, Healthy |

---

## 🧠 Model Architecture

A sequential CNN with five convolutional blocks, followed by a fully connected classifier. Total: **7,842,762 trainable parameters** (~30 MB).

| Stage | Layers |
|-------|--------|
| Input | RGB image resized to `128 × 128 × 3` |
| Block 1 | 2 × Conv2D (32 filters, 3×3, ReLU) → MaxPool (2×2) |
| Block 2 | 2 × Conv2D (64 filters, 3×3, ReLU) → MaxPool (2×2) |
| Block 3 | 2 × Conv2D (128 filters, 3×3, ReLU) → MaxPool (2×2) |
| Block 4 | 2 × Conv2D (256 filters, 3×3, ReLU) → MaxPool (2×2) |
| Block 5 | 2 × Conv2D (512 filters, 3×3, ReLU) → MaxPool (2×2) |
| Regularization | Dropout (0.25) |
| Classifier | Flatten → Dense (1500, ReLU) → Dropout (0.4) → Dense (38, Softmax) |

**Training configuration**

| Setting | Value |
|---------|-------|
| Optimizer | Adam (learning rate `0.0001`) |
| Loss | Categorical cross-entropy |
| Batch size | 32 |
| Epochs | 10 |
| Image size | 128 × 128 |
| Labels | One-hot (categorical), inferred from folder names |

A small learning rate (`0.0001`) is used to avoid overshooting the minimum, and extra convolution layers and dense units are added to help the model capture more features.

---

## 📊 Results

Training history is saved in [`training_hist.json`](training_hist.json).

| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|:-----:|:--------------:|:------------:|:----------:|:--------:|
| 1  | 60.2% | 85.0% | 1.348 | 0.485 |
| 2  | 86.0% | 91.6% | 0.437 | 0.263 |
| 3  | 91.7% | 93.3% | 0.258 | 0.209 |
| 4  | 94.1% | 92.5% | 0.179 | 0.227 |
| 5  | 95.6% | 94.8% | 0.132 | 0.167 |
| 6  | 96.7% | 95.8% | 0.103 | 0.133 |
| 7  | 97.2% | 96.5% | 0.088 | 0.110 |
| 8  | 97.6% | 95.5% | 0.072 | 0.157 |
| 9  | 98.0% | 95.6% | 0.061 | 0.148 |
| 10 | 98.2% | 95.3% | 0.056 | 0.174 |

**Final performance:** about **98% training accuracy** and **95% validation accuracy**.

The classification report on the 17,572 validation images (from the training notebook) shows **0.95 precision, recall and F1-score** (macro and weighted average). Most classes score 0.92 or higher. The weakest are `Tomato___Target_Spot` (F1 ≈ 0.89), `Potato___Late_blight` (precision 0.86) and `Apple___healthy` (recall 0.89).

---

## 🗂 Project Structure

```
Plant-Disease-Detection/
├── Train_plant_disease.ipynb   # Data loading, model building, training, evaluation
├── Test_Plant_Disease.ipynb    # Load saved model and predict a single image
├── training_hist.json          # Saved loss/accuracy history from training
├── requirement.txt             # Python dependencies
├── README.md
│
├── train/                      # (download separately) training images, one folder per class
├── valid/                      # (download separately) validation images, one folder per class
└── test/
    └── test/                   # (download separately) sample leaf images for testing
```

After training, `trained_model.keras` is also created in the project folder.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/surajit447/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

Main libraries: TensorFlow, scikit-learn, NumPy, pandas, Matplotlib, Seaborn. The test notebook also uses OpenCV:

```bash
pip install opencv-python
```

### 4. Download the dataset

Download the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle and arrange it so the `train`, `valid` and `test` folders sit next to the notebooks (see [Project Structure](#-project-structure)).

### 5. Train the model

Open and run **`Train_plant_disease.ipynb`**:

```bash
jupyter notebook Train_plant_disease.ipynb
```

This trains for 10 epochs, prints the evaluation metrics and saves `trained_model.keras`. Training on a CPU can be slow; a GPU is strongly recommended (roughly 4 minutes per epoch was observed on an Apple M2 with GPU acceleration).

### 6. Test the model

Open and run **`Test_Plant_Disease.ipynb`**. Before running, set `image_path` to a leaf image on your machine, for example:

```python
image_path = "test/test/TomatoEarlyBlight2.JPG"
```

> **Troubleshooting:** If you get errors saving or loading the `.keras` file, or about `tf.keras.optimizers.legacy`, install a newer TensorFlow (2.12 or later).

---

## 🔮 Making a Prediction

Once `trained_model.keras` exists, you can classify a leaf image in a few lines:

```python
import os
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("trained_model.keras")

# Class names in the same (alphabetical) order used during training
class_names = sorted(os.listdir("train"))

image = tf.keras.preprocessing.image.load_img("path/to/leaf.jpg", target_size=(128, 128))
x = np.expand_dims(tf.keras.preprocessing.image.img_to_array(image), axis=0)

probs = model.predict(x)[0]
print(f"Prediction: {class_names[np.argmax(probs)]}  ({probs.max():.1%} confidence)")
```

The image is only resized to 128 × 128. No extra normalization is applied, matching how the model was trained.

---

## ⚠️ Limitations

- **Dataset bias:** The images are mostly leaves photographed on plain backgrounds under controlled conditions. Accuracy may drop on real-world field photos with cluttered backgrounds, different lighting or multiple leaves.
- **Mild overfitting:** Training accuracy (98%) is higher than validation accuracy (95%), and validation loss rises slightly after epoch 7.
- **Fixed class set:** The model only knows the 38 classes listed above and will always return one of them, even for an unrelated image or an unsupported plant.
- **Validation used for reporting:** The reported metrics come from the validation set. The `test` folder contains only 33 unlabelled sample images.

This tool is for learning and research. It should not replace advice from an agricultural expert.

---

## 🛠 Future Improvements

- Add a Streamlit web app so users can upload a leaf photo and get an instant diagnosis.
- Use transfer learning (e.g. MobileNetV2, EfficientNet, ResNet) for higher accuracy with less training time.
- Add data augmentation, early stopping and learning-rate scheduling to reduce overfitting.
- Evaluate on real-world field images and a properly held-out labelled test set.
- Export the model to TensorFlow Lite for mobile use.
- Show disease descriptions and treatment suggestions with each prediction.

---


## 🎓 Academic Context

This project was developed as part of coursework at The Neotia University,
under the supervision of Dr. Joydeb Mondal, Department of Computer Science and Engineering.

---

## 🙏 Acknowledgements

- Dataset: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) on Kaggle, derived from the PlantVillage dataset.
- Built with [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/), [scikit-learn](https://scikit-learn.org/) and [Matplotlib](https://matplotlib.org/).

---

## 👤 Author

**surajit447** · [GitHub](https://github.com/surajit447)

If you found this project useful, consider giving it a ⭐.
