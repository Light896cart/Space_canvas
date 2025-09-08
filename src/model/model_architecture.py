import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        print(self.backbone)
        self.backbone.conv1 = nn.Conv2d(1,64,kernel_size=7)
        self.backbone.fc = nn.Linear(512, 3)

        # Или можно сделать так:
        # self.backbone.fc = nn.Sequential(
        #     nn.Dropout(0.5),  # опционально: регуляризация
        #     nn.Linear(512, num_classes)
        # )

    def forward(self, x):
        return self.backbone(x)

    def __repr__(self):
        return f'Это архитектура {self.backbone}'
