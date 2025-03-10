import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import json
import requests
# Load model
# Hugging Face Model URL (Replace with your actual model link)
MODEL_URL = "https://huggingface.co/AkhilNam/BreedDetector/resolve/main/fine_tuned_pet_mood_model.h5"
MODEL_PATH = "fine_tuned_pet_mood_model.h5"

# Check if model already exists, if not, download it
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    response = requests.get(MODEL_URL, stream=True)
    
    # Ensure the request was successful
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Model download complete!")
    else:
        raise Exception(f"Failed to download model, HTTP Status Code: {response.status_code}")

# Load the model
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

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

