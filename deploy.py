import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import json

# Load model
model_path = "fine_tuned_pet_mood_model.h5"  # Ensure this matches your saved model
model = load_model(model_path)


# Load breed namesimport json

# Load breed names from JSON file
with open("breed_names.json", "r") as f:
    breed_mapping = json.load(f)

print("Breed names loaded successfully!")


# Streamlit UI
st.title("🐶 Pet Mood Detector")

uploaded_file = st.file_uploader("Upload a Pet Image", type=["jpg", "png"])

if uploaded_file:
    # Save and process the image
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = load_img(file_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_breed = breed_mapping.get(predicted_class, "Unknown")

    # Display results
    st.image(img, caption=f"Predicted Breed: {predicted_breed}", use_column_width=True)
    st.success(f"Prediction: {predicted_breed}")

