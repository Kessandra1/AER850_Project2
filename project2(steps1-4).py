from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd    

# For reproducibility
np.random.seed(42)
keras.utils.set_random_seed(42)


# Step 1: Data Processing
print("\n-----Step 1: Data Processing-----")
# Define input image shape
width, height, channel = 500, 500, 3
# Establish training and validation directory (using relative paths)
train_dir = "Data/train"
valid_dir = "Data/valid"
# Perform Data augmentation (using Keras)
train_aug = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
)
valid_aug = ImageDataGenerator(rescale=1./255)
# Train and validation generators (flowfromdirectory made more sense)
BATCH_SIZE = 32 
train_generator = train_aug.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
valid_generator = valid_aug.flow_from_directory(
    valid_dir,
    target_size=(128, 128),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


# Step 2 & 3: Neural Network Architecture Design & Hyperparameter Analysis
print("\n-----Step 2 & 3: Neural Network Architecture Design & Hyperparameter Analysis-----")

# First Variation of DCNN model
mdl1 = models.Sequential([
    layers.Input(shape=(128, 128, channel)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])
# Parameter set for first model
mdl1.summary()
# Compilation of first model
mdl1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Second Variation of DCNN model
mdl2 = models.Sequential([
    layers.Input(shape=(128, 128, channel)),
    layers.Conv2D(64, (5,5), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (5,5), activation='relu6'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation=keras.layers.ELU(alpha=1.0)),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])
# Parameter set for second model
print("\n")
mdl2.summary()
# Compilation of second model
mdl2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Step 4: Model Evaluation
print("\n-----Step 4: Model Evaluation-----")

# Set Early Stop
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
    )
# Set size of epoch
EPOCHS = 10

# Store history of first DCNN model variation
history1 = mdl1.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)
# Plot Accuracy of first DCNN model variation
plt.figure(figsize=(6,4))
plt.plot(history1.history["accuracy"], label="Training Accuracy")
plt.plot(history1.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("1st DCNN Model Variation: Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
# Plot Loss of first DCNN model variation
plt.figure(figsize=(6,4))
plt.plot(history1.history["loss"], label="Training Loss")
plt.plot(history1.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("1st DCNN Model Variation: Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Store history of second DCNN model variation
print("\n")
history2 = mdl2.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)
# Plot Accuracy of second DCNN model variation
plt.figure(figsize=(6,4))
plt.plot(history2.history["accuracy"], label="Training Accuracy")
plt.plot(history2.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("2nd DCNN Model Variation: Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()
# Plot Loss of second DCNN model variation
plt.figure(figsize=(6,4))
plt.plot(history2.history["loss"], label="Training Loss")
plt.plot(history2.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("2nd DCNN Model Variation: Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()


# Save the better model variation
mdl2.save("test_model.keras")

