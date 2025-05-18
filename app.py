import os
import gdown

model_url = 'https://drive.google.com/file/d/1ZLXoTMb2Le_fvO4d9nQnvBoJgrOWlsUp/view?usp=sharing'  # Replace with actual FILE_ID
model_path = 'custom_cnn_best_model.keras'

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)



import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('custom_cnn_best_model.keras')
    return model

model = load_model()
CATEGORIES = ['fractured', 'non-fractured']

st.title('🦴 Bone Fracture X-ray Classifier')

uploaded_file = st.file_uploader("Upload a bone X-ray...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        st.write("⏳ Processing...")
        img = image.resize((224, 224)).convert('RGB')  # Ensure 3 channels
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

        prediction = model.predict(img_array)[0][0]
        label = CATEGORIES[int(prediction > 0.5)]
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.success(f'✅ Prediction: *{label}* with {confidence * 100:.2f}% confidence.')
