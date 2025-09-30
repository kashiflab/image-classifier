# 🐶🐱 Cats vs Dogs Image Classifier

This project is a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** to classify images of **cats vs dogs** using the Kaggle dataset.  

It is designed as a beginner-friendly project to understand:
- How to work with image datasets
- How CNNs work
- How to train, evaluate, and visualize model performance

---

## 📂 Project Structure
```

image-classifier/
│── venv/                # Virtual environment (not uploaded to GitHub)
│── dataset/             # Dogs & Cats images (downloaded via Kaggle)
│── train.py             # Training script
│── predict.py           # Script to test model predictions
│── model/               # Saved model (after training)
│── requirements.txt     # Required Python packages
│── README.md            # Project documentation

````

---

## ⚙️ Installation & Setup

### 1. Clone this repository
```bash
git clone https://github.com/kashiflab/image-classifier.git
cd image-classifier
````

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download dataset

This project uses the **Dogs vs Cats** dataset from Kaggle.

```python
import kagglehub

# Download latest version of dataset
path = kagglehub.dataset_download("chetankv/dogs-cats-images")
print("Path to dataset files:", path)
```

---

## 🚀 Training the Model

Run the training script:

```bash
python train.py
```

This will:

* Load and preprocess dataset
* Train CNN for 10 epochs
* Save trained model inside `model/`
* Show accuracy/loss graphs

---

## 🔎 Testing the Model

To test on new images:

```bash
python predict.py
```

Example output:

```
Image: dog1.jpg → Prediction: Dog (95%)
Image: cat3.jpg → Prediction: Cat (87%)
```

---

## 📊 Results

During training, the model reached:

* **Training Accuracy:** ~80%
* **Validation Accuracy:** ~81%

Example accuracy graph:

![Accuracy Graph](accuracy.png)

---

## 🧠 How It Works

1. **Dataset** → Cat & Dog images
2. **Preprocessing** → Resize images to 128×128, normalize pixel values (0–1)
3. **CNN Layers**

   * Convolution → Extract features (edges, shapes)
   * Pooling → Reduce image size, keep important info
   * Dense Layers → Learn combinations of features
   * Output → Predict Cat 🐱 or Dog 🐶

---

## 📦 Dependencies

Main packages used:

* `tensorflow` → Deep learning framework
* `matplotlib` → Plot training/validation graphs
* `opencv-python` → Image preprocessing
* `scipy` → Required by TensorFlow/Keras
* `kagglehub` → Download dataset from Kaggle

---

## 📈 Future Improvements

* Use **data augmentation** to improve generalization
* Train for more epochs
* Add **Dropout layers** for regularization
* Try **Transfer Learning** (e.g., MobileNet, ResNet) for higher accuracy (90%+)

---

## 🙌 Credits

* Dataset: [Dogs & Cats Images (Kaggle)](https://www.kaggle.com/datasets/chetankv/dogs-cats-images)
* Developed with ❤️ using TensorFlow & Python
