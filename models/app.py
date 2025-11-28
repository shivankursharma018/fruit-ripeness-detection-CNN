
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# -------------------------------
# Load TFLite Model
# -------------------------------
MODEL_PATH = "model.tflite"  # your tflite file

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# Class Labels (edit as needed)
# -------------------------------
CLASS_NAMES = [
    "Unripe", "Partially Ripe", "Ripe", 
    "Overripe", "Damaged", "Spoiled",
    "Rotten"
]

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_image(img):
    img = img.resize((64, 64))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------
# Prediction
# -------------------------------
def predict(img_tensor):
    interpreter.set_tensor(input_details[0]['index'], img_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0]

# -------------------------------
# UI
# -------------------------------
st.title("Fruit Ripeness Detection (TFLite on Windows)")

uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    img_tensor = preprocess_image(image)
    preds = predict(img_tensor)

    cls = np.argmax(preds)
    conf = float(np.max(preds)) * 100

    st.subheader(f"Prediction: **{CLASS_NAMES[cls]}**")
    st.write(f"Confidence: **{conf:.2f}%**")



