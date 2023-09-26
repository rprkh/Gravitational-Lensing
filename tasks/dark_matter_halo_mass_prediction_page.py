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
    MODEL_NAMES = ["efficientnet_b4", "convnext_base", "inception_resnet_v2"]
    DROPOUT = 0.3

class EfficientNetB4(nn.Module):
    def __init__(self, number_of_output_neurons):
        super(EfficientNetB4, self).__init__()
        
        self.effnetmodel = timm.create_model(CONFIG.MODEL_NAMES[0], pretrained=True, in_chans=1)
        
        for param in self.effnetmodel.parameters():
            param.requires_grad = True
            
        self.regressor = nn.Sequential(
            nn.Linear(1792 * 5 * 5, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=CONFIG.DROPOUT),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=CONFIG.DROPOUT),
            nn.Linear(64, number_of_output_neurons)
        )
                
    def forward(self, x):
        x = self.effnetmodel.forward_features(x)
        x = x.reshape(-1, 1792 * 5 * 5)
        x = self.regressor(x)
        
        return x

class Convnext_Base(nn.Module):
    def __init__(self, number_of_output_neurons):
        super(Convnext_Base, self).__init__()
        
        self.model = timm.create_model(CONFIG.MODEL_NAMES[1], pretrained=True, in_chans=1)
        
        for param in self.model.parameters():
            param.requires_grad = True
            
        self.regressor = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=CONFIG.DROPOUT),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=CONFIG.DROPOUT),
            nn.Linear(64, number_of_output_neurons)
        )
                
    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.reshape(-1, 1024 * 4 * 4)
        x = self.regressor(x)
        
        return x

class Inception_Resnet_V2(nn.Module):
    def __init__(self, number_of_output_neurons):
        super(Inception_Resnet_V2, self).__init__()
        
        self.model = timm.create_model(CONFIG.MODEL_NAMES[2], pretrained=True, in_chans=1)
        
        for param in self.model.parameters():
            param.requires_grad = True
            
        self.regressor = nn.Sequential(
            nn.Linear(1536 * 3 * 3, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=CONFIG.DROPOUT),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=CONFIG.DROPOUT),
            nn.Linear(64, number_of_output_neurons)
        )
        
    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.reshape(-1, 1536 * 3 * 3)
        x = self.regressor(x)
        
        return x

def get_model_options_halo_mass():
    model_types = ["EfficientNetB4", "ConvNeXtBase", "InceptionResNetV2"]
    selected_model = st.radio("Select a model", model_types)

    return selected_model