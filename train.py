"""
train.py
Loads data, builds the model, trains it, and saves the trained model.
"""

from data_loader import load_data, IMG_SIZE
from model import build_model
import matplotlib.pyplot as plt

# Load training, validation, test sets
train_gen, val_gen, test_gen = load_data()

# Build CNN model
model = build_model(img_size=IMG_SIZE)
model.summary()

# Train the model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# Evaluate on test set
loss, acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy: {acc:.2f}")

# Save model
model.save("cats_vs_dogs_model.h5")
print("💾 Model saved as cats_vs_dogs_model.h5")

# Plot training history
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.legend()
plt.show()
