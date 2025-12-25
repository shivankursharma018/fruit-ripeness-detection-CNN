
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import time
import pandas as pd



# Page Config
st.set_page_config(page_title="Fruit Ripeness Detection", layout="centered")

# Load Model
MODEL_PATH = "model.tflite"
CLASS_MAP_PATH = "class_names.json"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



# Load Correct Class Order
with open(CLASS_MAP_PATH, "r") as f:
    cls_map = json.load(f)

CLASS_NAMES = [None] * len(cls_map)
for name, idx in cls_map.items():
    CLASS_NAMES[idx] = name



# Preprocessing
def preprocess_image(img):
    img = img.resize((64, 64))
    img = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)



# Prediction
def predict(img_tensor):
    start_time = time.time()
    interpreter.set_tensor(input_details[0]["index"], img_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    inference_time = time.time() - start_time
    return output[0], inference_time

st.title("Fruit Ripeness Detection (Offline)")

with st.expander("Model Information"):
    st.write("Model format: TensorFlow Lite (.tflite)")
    st.write(f"Input shape: {input_details[0]['shape']}")
    st.write(f"Input dtype: {input_details[0]['dtype']}")
    st.write(f"Output shape: {output_details[0]['shape']}")
    st.write(f"Number of classes: {len(CLASS_NAMES)}")


uploaded_file = st.file_uploader(
    "Upload a fruit image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = preprocess_image(image)
    preds, inference_time = predict(img_tensor)

    # Prediction Result
    predicted_index = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    st.subheader("Prediction Result")
    st.write(f"Predicted Class: **{CLASS_NAMES[predicted_index]}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
    st.write(f"Inference Time: **{inference_time:.4f} seconds**")

    # Probability Distribution
    st.subheader("Class Probability Distribution")

    prob_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability (%)": np.round(preds * 100, 2)
    }).sort_values(
        by="Probability (%)",
        ascending=False
    ).reset_index(drop=True)

    st.table(prob_df)

    # top 3
    st.subheader("Top 3 Predictions")

    for i in range(3):
        st.write(
            f"{i+1}. {prob_df.loc[i, 'Class']} — "
            f"{prob_df.loc[i, 'Probability (%)']}%"
        )

