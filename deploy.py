import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import json
import requests
# Load model
MODEL_URL = "https://huggingface.co/AkhilNam/BreedDetector/resolve/main/fine_tuned_pet_mood_model.h5"
MODEL_PATH = "fine_tuned_pet_mood_model.h5"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

# Load the model
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

    # Print raw model predictions (for debugging)
    print("\n🔍 Raw Model Predictions:", predictions)

    # Extract predicted class index
    predicted_class = str(np.argmax(predictions, axis=1)[0])
    print("📌 Predicted Class Index:", predicted_class)

    # Verify breed mapping keys
    print("📂 Breed Mapping Keys (First 10):", list(breed_mapping.keys())[:10])
    print("📂 Breed Mapping Last 10 Keys:", list(breed_mapping.keys())[-10:])

    # Debug: Check if predicted class is in breed mapping
    if predicted_class in breed_mapping:
        predicted_breed = breed_mapping[predicted_class].replace("_", " ").title()
    else:
        print(f"🚨 WARNING: Predicted class {predicted_class} not in breed_mapping!")
        predicted_breed = "Unknown"

    print("✅ Final Predicted Breed:", predicted_breed)


    # Display results
    st.image(img, caption=f"Predicted Breed: {predicted_breed}", use_column_width=True)
    st.success(f"Prediction: {predicted_breed}")

