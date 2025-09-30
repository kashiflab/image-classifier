"""
predict.py
Loads the trained model and makes a prediction on a sample image.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

IMG_SIZE = 128

# Load the saved model
model = tf.keras.models.load_model("cats_vs_dogs_model.h5")

def predict_image(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)  # batch dimension

    # Make prediction
    prediction = model.predict(img_arr)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    print(f"Prediction: {label} (score={prediction:.4f})")

# Example usage
if __name__ == "__main__":
    predict_image("sample-images/sample1.jpeg")  # replace with a test image path
    predict_image("sample-images/sample2.jpeg")  # replace with a test image path
    predict_image("sample-images/sample3.jpeg")  # replace with a test image path
    predict_image("sample-images/sample4.jpeg")  # replace with a test image path
    predict_image("sample-images/sample5.jpeg")  # replace with a test image path
    predict_image("sample-images/sample6.jpeg")  # replace with a test image path
    predict_image("sample-images/sample7.jpeg")  # replace with a test image path
    predict_image("sample-images/sample8.jpeg")  # replace with a test image path
    predict_image("sample-images/sample9.jpeg")  # replace with a test image path
    predict_image("sample-images/sample10.jpeg")  # replace with a test image path
