import os
import json
import gdown
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

working_dir = os.path.dirname(os.path.abspath(__file__))


# Replace with your Google Drive File ID
FILE_ID = "1rSeyYG57iovB8RL6CDi2Wn__jHOKOQuX"

model_path = f"{working_dir}/plant_disease_model.keras"

# Download model if not present
if not os.path.exists(model_path):

    with st.spinner("Downloading model..."):

        url = f"https://drive.google.com/uc?id={FILE_ID}"

        gdown.download(url, model_path, quiet=False)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

class_indices = json.load(
    open(f"{working_dir}/class_indices.json")
)


def load_and_preprocess_image(image, target_size=(224, 224)):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize(target_size)

    # Convert image to numpy array
    img_array = np.array(image)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize image
    img_array = img_array.astype("float32") / 255.0

    return img_array

# =========================
# PREDICTION FUNCTION
# =========================

def predict_image_class(model, image, class_indices):

    preprocessed_img = load_and_preprocess_image(image)

    predictions = model.predict(preprocessed_img)

    predicted_class_index = np.argmax(predictions)

    confidence = np.max(predictions) * 100

    predicted_class_name = class_indices.get(
        str(predicted_class_index),
        "Unknown"
    )

    return predicted_class_name, confidence

st.title("🌿 Plant Disease Classifier")

st.write(
    "Upload a plant leaf image to detect plant diseases."
)

uploaded_image = st.file_uploader(
    "Upload an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:

    image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)

    with col1:

        resized_img = image.resize((250, 250))

        st.image(
            resized_img,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        st.write("### Prediction")

        if st.button('Classify'):

            with st.spinner("Analyzing image..."):

                prediction, confidence = predict_image_class(
                    model,
                    image,
                    class_indices
                )

                st.success(f"Prediction: {prediction}")

                st.info(f"Confidence: {confidence:.2f}%")