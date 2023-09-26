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
    MODEL_NAMES = ["vit_base_patch16_224"]
    DROPOUT = 0.3


class ViT_Base_Patch_16_224(nn.Module):
    def __init__(self, n_classes):
        super(ViT_Base_Patch_16_224, self).__init__()

        self.vit_model = timm.create_model(
            CONFIG.MODEL_NAMES[0], pretrained=True, in_chans=3
        )

        for param in self.vit_model.parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(197 * 768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=CONFIG.DROPOUT),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=CONFIG.DROPOUT),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.vit_model.forward_features(x)
        x = x.view(-1, 197 * 768)
        x = self.classifier(x)

        return x


def get_model_options():
    model_types = ["ViT_Base_Patch_16_224"]
    selected_model = st.radio("Select a model", model_types)

    return selected_model
