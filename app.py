
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# Application title
st.title("MNIST Handwritten Digit Recognition")

st.write(
    "Upload an image of a handwritten digit "
    "and let our trained CNN model predict the number."
)

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn.keras")

model = load_model()

# Image upload
uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, caption="Your uploaded digit", width=200)

    if st.button("Predict Digit"):

        # Convert image to grayscale
        image = ImageOps.grayscale(image)

        # Resize image to 28x28
        image = ImageOps.invert(image)
        image = image.resize((28, 28))

        # Convert image to array
        image_array = np.array(image).astype("float32")

        # Normalize pixel values
        image_array = image_array / 255.0

        # Reshape for CNN model
        image_array = image_array.reshape(1, 28, 28, 1)

        # Make prediction
        predictions = model.predict(image_array, verbose=0)

        predicted_digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions)) * 100

        # Display results
        st.subheader("Prediction Result")

        st.success(f"Predicted Digit: {predicted_digit}")

        st.write(f"Prediction Confidence: {confidence:.2f}%")

        st.write("Prediction probabilities:")

        for digit, probability in enumerate(predictions[0]):
            st.write(f"Digit {digit}: {probability * 100:.2f}%")
