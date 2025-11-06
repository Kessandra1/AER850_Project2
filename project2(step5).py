from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np 

# For reproducibility
np.random.seed(42)
keras.utils.set_random_seed(42)


# Step 5: Model Testing
print("\n-----Step 5: Model Testing-----")

# Load trained model
trained_mdl = load_model("test_model.keras")


test_images = {
    "crack": "Data/test/crack/test_crack.jpg",
    "missing-head": "Data/test/missing-head/test_missinghead.jpg",
    "paint-off": "Data/test/paint-off/test_paintoff.jpg"
}

# Preprocess and predict the image class
for true_label, img_path in test_images.items():
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred_prob = trained_mdl.predict(img_array)
    pred_class = np.argmax(pred_prob, axis=1)[0]
    class_indices = {"crack":0, "missing-head":1, "paint-off":2}
    idx_to_class = {v:k for k,v in class_indices.items()}
    pred_label = idx_to_class[pred_class]

    plt.imshow(img)
    plt.axis("off")
    color = "green" if pred_label == true_label else "red"
    plt.title(f"Predicted:{pred_label} / True:{true_label}", color=color)
    plt.tight_layout()
    plt.show()
