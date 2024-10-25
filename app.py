import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import joblib
import cv2  # Ensure cv2 is imported for HOG feature extraction
from skimage.feature import hog  # Ensure hog is imported

# Load the trained model
loaded_model = joblib.load('random_forest_model.pkl')
# Set image dimensions
IMG_HEIGHT = 180
IMG_WIDTH = 180

# Define class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Replace with actual class names

def extract_hog_features(image):
    """Extract HOG features from the image."""
    # Ensure the image is in the correct format
    image = (image * 255).astype(np.uint8)  # Convert from float64 (0 to 1) back to uint8 (0 to 255)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute HOG features
    features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                   orientations=9, block_norm='L2-Hys', visualize=False)
    return features

def predict_image(image):
    """Load, preprocess the image, and make predictions."""
    image_array = np.array(image) / 255.0  # Rescale
    features = extract_hog_features(image_array) 
    features = features.reshape(1, -1)  

    # Make prediction
    prediction = loaded_model.predict(features)
    predicted_class = class_names[prediction[0]]  
    return predicted_class

# Streamlit UI
st.title("Image Classification with SVM")
st.write("Upload one or more images to get their predicted classes.")

# File uploader for multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files is not None:
    # Prediction button for all images
    if st.button("Predict"):
        for uploaded_file in uploaded_files:
            # Load the image for prediction
            image = load_img(uploaded_file, target_size=(IMG_HEIGHT, IMG_WIDTH))
            prediction_result = predict_image(image)

            # Display the predicted image with its class
            st.image(image, caption=f'Predicted Class: {prediction_result}', use_column_width=True)
