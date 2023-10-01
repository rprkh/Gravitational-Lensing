import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import platform
import torch
import torch.nn.functional as F
from tasks.binary_substructure_classification_page import (
    get_model_options_binary_classification,
    ViT_Base_Patch_16_224,
)
from tasks.dark_matter_halo_mass_prediction_page import (
    get_model_options_halo_mass,
    EfficientNetB4,
    Convnext_Base,
    Inception_Resnet_V2,
)
from tasks.multiclass_substructure_classification_page import get_model_options_multiclass_classification, TransferLearningModelNew, DenseNet201, MobileVitV2_150, DenseNetEnsemble

st.title("Gravitational Lensing")


def get_device():
    global DEVICE
    if torch.cuda.is_available():
        message = "`[INFO]` Using **GPU**: *{}*\n".format(torch.cuda.get_device_name())
        DEVICE = torch.device("cuda:0")
    else:
        message = "\n`[INFO]` GPU not found. Using **CPU**: *{}*\n".format(
            platform.processor()
        )
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

    model.eval()

    with torch.no_grad():
        y_pred = model(image.float())
        _, predicted = torch.max(y_pred.data, 1)
        class_label = predicted.item()
        model_prediction = f"**Model Prediction**: {class_label}"

        if class_label == 0:
            class_prediction = "**Class**: No Substructure"
        if class_label == 1:
            class_prediction = "**Class**: Substructure"

        probs = F.softmax(y_pred, dim=1)
        final_prob = torch.max(probs)
        confidence = f"**Confidence**: {round(final_prob.item() * 100, 2)}%"

    return model_prediction, class_prediction, confidence


def dark_matter_halo_mass_prediction_image_processing(image):
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    image = torch.tensor(image)
    image = image.to(DEVICE)

    return image


def dark_matter_halo_mass_prediction_regression_results(model, image):
    model.eval()

    with torch.no_grad():
        y_pred = model(image.float())
        y_pred = y_pred.view(-1)
        y_pred = y_pred.type(torch.float64)
        model_prediction = f"**Dark Matter Halo Mass**: {y_pred.item()}"

    return model_prediction

def multiclass_substructure_classification_image_processing(image):
    image = np.expand_dims(image, axis=0)
    image = np.transpose(image, (0, 3, 1, 2))
    image = (torch.tensor(image)).to(DEVICE)

    return image


def multiclass_substructure_clasification_results(model, image):
    image = torch.tensor(image)
    image = image.to(DEVICE)

    model.eval()

    with torch.no_grad():
        y_pred = model(image.float())
        _, predicted = torch.max(y_pred.data, 1)
        class_label = predicted.item()
        model_prediction = f"**Model Prediction**: {class_label}"

        if class_label == 0:
            class_prediction = "**Class**: No Substructure"
        if class_label == 1:
            class_prediction = "**Class**: Vortex Substructure"
        if class_label == 2:
            class_prediction = "**Class**: Sphere Substructure"

        probs = F.softmax(y_pred, dim=1)
        final_prob = torch.max(probs)
        confidence = f"**Confidence**: {round(final_prob.item() * 100, 2)}%"

    return model_prediction, class_prediction, confidence

st.sidebar.title("Navigation")
uploaded_image = st.sidebar.file_uploader(
    "Upload your image", type=["png", "jpg", "jpeg", "npy"]
)

type_of_task = [
    "Binary Substructure Classification",
    "Dark Matter Halo Mass Prediction",
    "Multiclass Substructure Classification",
]
selected_option = st.sidebar.radio("Select a task:", type_of_task)

if uploaded_image is not None:
    file_name = uploaded_image.name
    if file_name.endswith((".jpg", ".jpeg", ".png")):
        try:
            file_bytes = np.asarray(bytearray(uploaded_image.read()))
            opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except:
            st.write("***Predictions for this image are not possible. Please upload another image***")

    if file_name.endswith((".npy")):
        try:
            if selected_option == type_of_task[1]:
                image, _ = np.load(uploaded_image, allow_pickle=True)
            if selected_option == type_of_task[2]:
                image = np.load(uploaded_image, allow_pickle=True)
                image = image.swapaxes(0, 1)
                image = image.swapaxes(1, 2)
            fig, ax = plt.subplots()
            ax.imshow(image)
            plt.axis("off")
            st.pyplot(fig)

            if selected_option == type_of_task[1]:
                image = np.expand_dims(image, axis=2)
        except:
            st.write("***Predictions for this image are not possible. Please upload another image***")

    message = get_device()

    if selected_option == type_of_task[0]:
        selected_model = get_model_options_binary_classification()
        st.write(message)
        try:
            image = binary_substructure_classification_image_processing(opencv_image)

            if selected_model == "ViT_Base_Patch_16_224":
                vit_model = ViT_Base_Patch_16_224(2)
                vit_model = vit_model.to(DEVICE)
                vit_model.load_state_dict(
                    torch.load(
                        "models/binary_substructure_classification/vit_base_patch16_224_epochs_20_batchsize_64_lr_0.0001.pth",
                        map_location=DEVICE,
                    )
                )
                st.write("*Model loaded successfully*")

                (
                    model_prediction,
                    class_prediction,
                    confidence,
                ) = binary_substructure_classification_results(vit_model, image)
                st.write(model_prediction)
                st.write(class_prediction)
                st.write(confidence)
        except:
            st.write("***Predictions for this image are not possible. Please upload another image***")

    if selected_option == type_of_task[1]:
        selected_model = get_model_options_halo_mass()
        st.write(message)
        try:
            if selected_model == "EfficientNetB4":
                image = dark_matter_halo_mass_prediction_image_processing(image)

                efficient_net_b4_model = EfficientNetB4(1)
                efficient_net_b4_model = efficient_net_b4_model.to(DEVICE)
                efficient_net_b4_model.load_state_dict(
                    torch.load(
                        "models/dark_matter_halo_mass_prediction/efficientnet_b4_epochs_10_batchsize_128_lr_0.0005.pth",
                        map_location=DEVICE,
                    )
                )
                st.write("Model loaded successfully")

                model_prediction = dark_matter_halo_mass_prediction_regression_results(
                    efficient_net_b4_model, image
                )
                st.write(model_prediction)

            if selected_model == "ConvNeXtBase":
                image = dark_matter_halo_mass_prediction_image_processing(image)

                convnext_base_model = Convnext_Base(1)
                convnext_base_model = convnext_base_model.to(DEVICE)
                convnext_base_model.load_state_dict(
                    torch.load(
                        "models\dark_matter_halo_mass_prediction/convnext_base_epochs_25_batchsize_128_lr_5e-05.pth",
                        map_location=DEVICE,
                    )
                )
                st.write("Model loaded successfully")

                model_prediction = dark_matter_halo_mass_prediction_regression_results(
                    convnext_base_model, image
                )
                st.write(model_prediction)

            if selected_model == "InceptionResNetV2":
                image = dark_matter_halo_mass_prediction_image_processing(image)

                inception_resnet_v2_model = Inception_Resnet_V2(1)
                inception_resnet_v2_model = inception_resnet_v2_model.to(DEVICE)
                inception_resnet_v2_model.load_state_dict(
                    torch.load(
                        "models/dark_matter_halo_mass_prediction/inception_resnet_v2_epochs_20_batchsize_128_lr_5e-05.pth",
                        map_location=DEVICE,
                    )
                )
                st.write("Model loaded successfully")

                model_prediction = dark_matter_halo_mass_prediction_regression_results(
                    inception_resnet_v2_model, image
                )
                st.write(model_prediction)
        except:
            st.write("***Predictions for this image are not possible. Please upload another image***")

    if selected_option == type_of_task[2]:
        selected_model = get_model_options_multiclass_classification()
        st.write(message)

        image = multiclass_substructure_classification_image_processing(image)

        if selected_model == "DenseNet161":
            densenet161 = TransferLearningModelNew(3)
            densenet161 = densenet161.to(DEVICE)
            densenet161.load_state_dict(
                torch.load(
                    "models/multiclass_substructure_classification/densenet161_epochs_15_batchsize_64_lr_0.0001.bin",
                    map_location=DEVICE,
                )
            )
            st.write("Model loaded successfully")

            model_prediction, class_prediction, confidence = multiclass_substructure_clasification_results(densenet161, image)
            st.write(model_prediction)
            st.write(class_prediction)
            st.write(confidence)

        if selected_model == "DenseNet201":
            densenet201 = DenseNet201(3)
            densenet201 = densenet201.to(DEVICE)
            densenet201.load_state_dict(
                torch.load(
                    "models/multiclass_substructure_classification/densenet201_epochs_15_batchsize_64_lr_0.0001.bin",
                    map_location=DEVICE
                )
            )
            st.write("Model loaded successfully")

            model_prediction, class_prediction, confidence = multiclass_substructure_clasification_results(densenet201, image)
            st.write(model_prediction)
            st.write(class_prediction)
            st.write(confidence)

        if selected_model == "MobileVitV2_150_384_in22ft1k":
            mobile_vit = MobileVitV2_150(3)
            mobile_vit = mobile_vit.to(DEVICE)
            mobile_vit.load_state_dict(
                torch.load(
                    "models/multiclass_substructure_classification/mobilevitv2_150_epochs_15_batchsize_32_lr_0.0001.bin",
                    map_location=DEVICE
                )
            )
            st.write("Model loaded successfully")

            model_prediction, class_prediction, confidence = multiclass_substructure_clasification_results(mobile_vit, image)
            st.write(model_prediction)
            st.write(class_prediction)
            st.write(confidence)

        if selected_model == "DenseNet Ensemble":
            densenet_ensemble = DenseNetEnsemble(3, TransferLearningModelNew(3).to(DEVICE), DenseNet201(3).to(DEVICE))
            densenet_ensemble = densenet_ensemble.to(DEVICE)
            densenet_ensemble.load_state_dict(
                torch.load(
                    "models/multiclass_substructure_classification/ensemble_epochs_10_batchsize_32_lr_0.0001.bin",
                    map_location=DEVICE
                )
            )
            st.write("Model loaded successfully")

            model_prediction, class_prediction, confidence = multiclass_substructure_clasification_results(densenet_ensemble, image)
            st.write(model_prediction)
            st.write(class_prediction)
            st.write(confidence)
else:
    st.write("Please upload an image")
