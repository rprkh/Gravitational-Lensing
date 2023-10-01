import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import timm
from dataclasses import dataclass
import streamlit as st

@dataclass
class CONFIG:
    MODEL_NAMES = ['densenet161', 'mobilevitv2_150', 'mobilevitv2_150_384_in22ft1k']
    DROPOUT = 0.3

class TransferLearningModelNew(nn.Module):
    def __init__(self, n_classes):
        super(TransferLearningModelNew, self).__init__()
        self.transfer_learning_model = timm.create_model(CONFIG.MODEL_NAMES[0], pretrained=True, in_chans=1)
        
        for param in self.transfer_learning_model.parameters():
            param.requires_grad = True
            
        self.classifier = nn.Sequential(
            nn.Linear(2208 * 4 * 4, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.33),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.33),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x):
        x = self.transfer_learning_model.forward_features(x)
        x = x.view(-1, 2208 * 4 * 4)
        x = self.classifier(x)
        
        return x

class DenseNet201(nn.Module):
    def __init__(self, n_classes):
        super(DenseNet201, self).__init__()
        self.transfer_learning_model = timm.create_model("densenet201", pretrained=True, in_chans=1)
        
        for param in self.transfer_learning_model.parameters():
            param.requires_grad = True
            
        self.classifier = nn.Sequential(
            nn.Linear(1920 * 4 * 4, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.33),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.33),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x):
        x = self.transfer_learning_model.forward_features(x)
        x = x.view(-1, 1920 * 4 * 4)
        x = self.classifier(x)
        
        return x
    
class MobileVitV2_150(nn.Module):
    def __init__(self, n_classes):
        
        super(MobileVitV2_150, self).__init__()
        
        self.vit_model = timm.create_model(CONFIG.MODEL_NAMES[2], pretrained=True, in_chans=1)
        
        for param in self.vit_model.parameters():
            param.requires_grad = True
            
        self.classifier = nn.Sequential(
            nn.Linear(768 * 6 * 6, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=CONFIG.DROPOUT),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=CONFIG.DROPOUT),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x):
        x = self.vit_model.forward_features(x)
        x = x.reshape(-1, 768 * 6 * 6)
        x = self.classifier(x)
        
        return x
    
class DenseNetEnsemble(nn.Module):
    def __init__(self, n_classes, modela, modelb):
        
        super(DenseNetEnsemble, self).__init__()
        
        self.modela = modela
        self.modelb = modelb
        
    def forward(self, x):
        outa = self.modela(x)
        outb = self.modelb(x)
        out = outa + outb
        x = out
        
        return x

def get_model_options_multiclass_classification():
    model_types = ["DenseNet161", "DenseNet201", "MobileVitV2_150_384_in22ft1k", "DenseNet Ensemble"]
    selected_model = st.radio("Select a model", model_types)

    return selected_model

DEVICE = torch.device("cpu")

model3 = TransferLearningModelNew(3)
model3 = model3.to(DEVICE)
model3.load_state_dict(torch.load(f"models/multiclass_substructure_classification/densenet161_epochs_15_batchsize_64_lr_0.0001.bin", map_location=torch.device("cpu")))
print("Model loaded succesfully")

model4 = MobileVitV2_150(3)
model4 = model4.to(DEVICE)
model4.load_state_dict(torch.load('models/multiclass_substructure_classification/mobilevitv2_150_epochs_15_batchsize_32_lr_0.0001.bin', map_location=torch.device("cpu")))
print("Model loaded succesfully")

model5 = DenseNet201(3)
model5 = model5.to(DEVICE)
model5.load_state_dict(torch.load('models/multiclass_substructure_classification/densenet201_epochs_15_batchsize_64_lr_0.0001.bin', map_location=torch.device("cpu")))
print("Model loaded succesfully")

modela = TransferLearningModelNew(3)
modela = modela.to(DEVICE)

modelb = DenseNet201(3)
modelb = modelb.to(DEVICE)

model = DenseNetEnsemble(3, modela, modelb)
model = model.to(DEVICE)
model.load_state_dict(torch.load(f'models/multiclass_substructure_classification/ensemble_epochs_10_batchsize_32_lr_0.0001.bin', map_location=torch.device("cpu")))
print("Model loaded succesfully")