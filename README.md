# Gravitational Lensing

Dark matter is a hypothetical form of matter, which does not interact with electromagnetic radiation. It does not reflect or emit light and is not directly observable by the human eye. However, its existence can be inferred by observing gravitational effects on visible matter, such as stars and galaxies. Gravitational lensing causes light to bend in the presence of a strong gravitational field. Due to this bending of light, distant objects may appear to be distorted or magnified. The study of these distorted shapes can aid researchers in identifying the distribution and location of dark matter. By analysing a variety of different images, it is possible to deduce the distribution of dark matter. Furthermore, by measuring the distortion geometry, the mass of the surrounding cluster of dark matter can be determined. This project performs 3 fundamental tasks related to dark matter and gravitational lensing:
- Binary Substructure Classification
- Dark Matter Halo Mass Prediction
- Multiclass Substructure Classification

# Binary Substructure Classification

| Deep Learning Model                                 | Epochs | Batch Size | Learning Rate | ROC AUC (OVO) | ROC AUC (OVR) |
| :-------------------------------------------------- | :----- | :--------- | :------------ | :------------ | :------------ |
| ViT_Base_Patch_16_224                               | 20     | 64         | 0.0001        | 0.99800       | 0.99800       |

### Results for ViT_Base_Patch_16_224 (20 Epochs):

![image](https://user-images.githubusercontent.com/75483881/224233826-e5d5fc32-f9ca-4be6-8e0d-7cb3c728828b.png)

![image](https://user-images.githubusercontent.com/75483881/224234139-8ecf59c8-f3a4-40dc-8ee2-d1d4f447f3f5.png)

# Dark Matter Halo Mass Prediction

### Models Used:

| Deep Learning Model                                 | Epochs | Batch Size | Learning Rate | MSE       |
| :-------------------------------------------------- | :----- | :--------- | :------------ | :-------- |
| EfficientNetB4                                      | 10     | 128        | 0.0005        | 0.0002007 |
| ConvNeXtBase                                        | 25     | 128        | 5e-05         | 0.0002763 |
| InceptionResNetV2                                   | 20     | 128        | 5e-05         | 0.0002618 |

### Results for EfficientNetB4 (10 Epochs):

![image](https://user-images.githubusercontent.com/75483881/227117195-78bbe109-c16f-42bb-8574-0fbfc96f0347.png)

### Results for ConvNeXtBase (25 Epochs):

![image](https://user-images.githubusercontent.com/75483881/227117096-0063fba6-f38c-4e77-805e-b50a6639fabf.png)

### Results for InceptionResNetV2 (20 Epochs):

![image](https://user-images.githubusercontent.com/75483881/227116960-8d363a39-6efd-4a00-977e-2e7f36c89ee1.png)

# Multiclass Substructure Classification

### Models Used:

| Deep Learning Model                         | Epochs | Batch Size | Learning Rate | ROC AUC (OVO) | ROC AUC (OVR) |
| :------------------------------------------ | :----- | :--------- | :------------ | :------------ | :------------ |
| DenseNet161                                 | 15     | 64         | 0.0001        | 0.98          | 0.98      |
| MobileVitV2_150_384_in22ft1k                | 15     | 32         | 0.0001        | 0.95          | 0.95      |
| DenseNet201                                 | 15     | 64         | 0.0001        | 0.97          | 0.97      |
| Ensemble_DenseNet161_DenseNet201            | 10     | 32         | 0.0001        | 0.98          | 0.98      |

### Results for DenseNet161 (15 Epochs):

![image](https://user-images.githubusercontent.com/75483881/224229172-4de710e2-7d15-4628-8510-1c6381abdd0a.png)

![image](https://user-images.githubusercontent.com/75483881/224229506-660f53ce-362f-4f89-9ef8-5f14d07d4d18.png)

### Results for MobileVitV2_150_384_in22ft1k (15 Epochs):

![image](https://user-images.githubusercontent.com/75483881/224229270-38789ff1-5fa3-4357-8687-f517dc4565b8.png)

![image](https://user-images.githubusercontent.com/75483881/224229544-f1b38158-0d4d-48c0-8507-61496a5c3d72.png)

### Results for DenseNet201 (15 Epochs):

![image](https://user-images.githubusercontent.com/75483881/224229327-4a6c5445-7bda-440b-a768-2d201f2c3c23.png)

![image](https://user-images.githubusercontent.com/75483881/224229574-ef27eb39-e5f0-459f-a338-7d3339db6286.png)

### Results for Ensemble Model (DenseNet161 & DenseNet201 for 10 Epochs):

![image](https://user-images.githubusercontent.com/75483881/224229401-9bf29204-0551-45d1-81ec-e91f1138ea8a.png)

![image](https://user-images.githubusercontent.com/75483881/224229611-0409d10c-6807-4de4-96f9-4d97ffdcb6b2.png)

# Usage

Clone the repository

```bash
https://github.com/rprkh/Gravitational-Lensing.git
```

Navigate to the root directory of the project
```bash
cd Gravitational-Lensing
```

Install the requirements
```bash
pip install -r requirements.txt
```

Run the following command
```bash
mkdir "models\binary_substructure_classification" "models\dark_matter_halo_mass_prediction" "models\multiclass_substructure_classification"
```

Download the trained models from the following Google Drive link: 
https://drive.google.com/drive/folders/1BBHnK_1K-DZ2Der2fWB6ABIKSDN5sCLr?usp=sharing

Add these models to their respective folders within the `models` directory of the project

Execucte the following command
```bash
streamlit run streamlit_app.py
```

The streamlit application should start on `http://localhost:8501/`
