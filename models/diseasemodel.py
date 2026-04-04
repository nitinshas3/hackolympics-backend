import tensorflow as tf
import numpy as np
import json
from pathlib import Path

IMG_SIZE = 240

# Use absolute paths relative to project root
BASE_DIR = Path(__file__).parent.parent  # Points to Hackolympics/ root

CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
WEIGHTS_PATH = BASE_DIR / "models" / "efficientnetb1_plant_final.weights.h5"

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

# Build model architecture
base_model = tf.keras.applications.EfficientNetB1(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
])

# Load weights
model.load_weights(str(WEIGHTS_PATH))
print(f"✅ EfficientNetB1 Plant Disease Model Loaded Successfully!")
print(f"   Classes: {len(CLASS_NAMES)}")
print(f"   Weights path: {WEIGHTS_PATH}")


def predict_disease(image_bytes: bytes) -> dict:
    """Predict disease from raw image bytes (used by FastAPI)"""
    # Decode and preprocess
    img = tf.image.decode_image(image_bytes, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.expand_dims(img, 0) / 255.0

    predictions = model.predict(img, verbose=0)[0]

    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])
    label = CLASS_NAMES[class_index]

    return {
        "disease": label,
        "confidence": round(confidence * 100, 2),
        "is_healthy": "healthy" in label.lower()
    }


# ==================== Test Block (Run directly) ====================
if __name__ == "__main__":
    test_image_path = BASE_DIR / "assets" / "Pepper__bell___Bacterial_spot.jpg"

    if test_image_path.exists():
        with open(test_image_path, "rb") as f:
            image_bytes = f.read()

        result = predict_disease(image_bytes)
        print("\nTest Prediction:")
        print(result)
    else:
        print(f"Test image not found at: {test_image_path}")