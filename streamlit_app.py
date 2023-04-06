import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the Teachable Machine model
model_path = "path/to/your/downloaded/model"
model = tf.keras.models.load_model(model_path)

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Preprocess the image for the model
        img = Image.fromarray(img)
        img_resized = img.resize((224, 224))  # Resize the image according to your model's input size
        img_array = np.array(img_resized)
        img_array = img_array / 255.0
        img_array = img_array[np.newaxis, ...]

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=-1)[0]

        # Add the predicted class label to the image
        cv2.putText(img, f"Predicted class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return img

st.title("Teachable Machine Streamlit App")
st.subheader("Using webcam input")

# Start the webcam
webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if not webrtc_ctx.video_transformer:
    st.write("Waiting for webcam input...")

# Run the app
if __name__ == "__main__":
    st.run()
