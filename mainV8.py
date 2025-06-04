import streamlit as st
import cv2 as cv
import numpy as np
import keras

# Class labels (3 for each possible output)
label_name = ['Bacterial Leaf Blight / Hawar Daun', 'Brown Spot / Bercak Daun', 'Healthy / Sehat','Leaf Blast', 'Tungro'] 

st.write("""
# Rice Leaf Disease Detection
This model is trained to detect various rice leaf diseases. Upload a clear image of a rice leaf for disease detection.
""")

model = keras.models.load_model('Training/rice_leaf_disease_model8.h5') 

uploaded_file = st.file_uploader("Upload an image of a rice leaf")
if uploaded_file is not None:
    # Read image, process image
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (224, 224)), axis=0) / 255.0

    # Prediction
    predictions = model.predict(normalized_image)
    st.image(image_bytes)

    # Result
    confidence = predictions[0][np.argmax(predictions)] * 100
    if confidence >= 80:
        st.write(f"Result: {label_name[np.argmax(predictions)]} (Confidence: {confidence:.2f}%)")
    else:
        st.write("Upload a clearer image.")
