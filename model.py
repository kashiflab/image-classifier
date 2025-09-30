"""
model.py
Defines the CNN model architecture for Cats vs Dogs classification.
"""

from tensorflow.keras import layers, models

def build_model(img_size=128):
    model = models.Sequential([
        # First Conv layer: 32 filters of size 3x3
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(img_size, img_size, 3)),
        layers.MaxPooling2D(2,2),  # reduce spatial size

        # Second Conv layer: 64 filters
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        # Third Conv layer: 128 filters
        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        # Flatten + Dense layers
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),  # prevent overfitting
        layers.Dense(1, activation="sigmoid")  # binary classification output
    ])

    # Compile model (Adam optimizer, Binary Crossentropy loss)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
