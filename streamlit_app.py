import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import platform
import torch
import torch.nn.functional as F
from tasks.binary_substructure_classification_page import get_model_options, ViT_Base_Patch_16_224
from tasks.binary_substructure_classification_page import Swin_Base_Patch4_Window7_224, SwinEnsemble
from tasks.binary_substructure_classification_page import Swin_s3_Base_224, Swinv2_cr_Base_224

st.title("Gravitational Lensing")

def get_device():
    global DEVICE
    if torch.cuda.is_available():
        message = "`[INFO]` Using **GPU**: *{}*\n".format(torch.cuda.get_device_name())
        DEVICE = torch.device("cuda:0")
    else:
        message = "\n`[INFO]` GPU not found. Using **CPU**: *{}*\n".format(platform.processor())
        DEVICE = torch.device("cpu")

    return message

def binary_substructure_classification_image_processing(image):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))

    return image

def binary_substructure_classification_results(model, image):
    image = torch.tensor(image)
    image = image.to(DEVICE)

    with torch.no_grad():
        model.eval()
        y_pred = model(image.float())
        _, predicted = torch.max(y_pred.data, 1)
        class_label = predicted.item()
        model_prediction = f"**Model Prediction**: {class_label}"

        if class_label == 0:
            class_prediction = "**Class**: No-Substructure"
        if class_label == 1:
            class_prediction = "**Class**: Substructure"

        probs = F.softmax(y_pred, dim=1)
        final_prob = torch.max(probs)
        confidence = f"**Confidence**: {round(final_prob.item() * 100, 2)}%"

        return model_prediction, class_prediction, confidence


st.sidebar.title("Navigation")
uploaded_image = st.sidebar.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])

type_of_task = [
    "Binary Substructure Classification",
    "Dark Matter Halo Mass Prediction",
    "Multiclass Substructure Classification",
]
selected_option = st.sidebar.radio("Select a task:", type_of_task)

if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()))
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # st.write(opencv_image.shape )

    image = Image.open(uploaded_image)
    st.image(image.resize((250, 250)), caption="Uploaded Image", use_column_width=False)

    message = get_device()

    if selected_option == type_of_task[0]:
        selected_model = get_model_options()
        st.write(message)

        image = binary_substructure_classification_image_processing(opencv_image)
        
        if selected_model == "ViT_Base_Patch_16_224":
            vit_model = ViT_Base_Patch_16_224(2)
            vit_model = vit_model.to(DEVICE)
            vit_model.load_state_dict(torch.load("models/vit_base_patch16_224_epochs_20_batchsize_64_lr_0.0001.pth", map_location=torch.device("cpu")))
            st.write("*Model loaded successfully*")

            model_prediction, class_prediction, confidence = binary_substructure_classification_results(vit_model, image)
            st.write(model_prediction)
            st.write(class_prediction)
            st.write(confidence)
else:
    st.write("Please upload an image")
