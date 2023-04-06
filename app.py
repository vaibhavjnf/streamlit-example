import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the Teachable Machine model
model_path = "path/to/your/downloaded/model"
model = tf.keras.models.load_model(model_path)
