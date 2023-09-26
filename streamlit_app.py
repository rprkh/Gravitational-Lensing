import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import platform
import torch
from tasks.binary_substructure_classification_page import BSC

st.title("Gravitational Lensing")

def get_device():
    global DEVICE
    if torch.cuda.is_available():
        message = "[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name())
        DEVICE = torch.device("cuda:0")
    else:
        message = "\n[INFO] GPU not found. Using CPU: {}\n".format(platform.processor())
        DEVICE = torch.device("cpu")

    return message

st.sidebar.title("Navigation")
uploaded_image = st.sidebar.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])

type_of_task = [
    "Binary Substructure Classification",
    "Dark Matter Halo Mass Prediction",
    "Multiclass Substructure Classification",
]
selected_option = st.sidebar.radio("Select an option:", type_of_task)

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

message = get_device()
# st.write(message)

if selected_option == type_of_task[0]:
    st.write(message)
    st.write("Binary")
    m=BSC()
    st.write(m)
