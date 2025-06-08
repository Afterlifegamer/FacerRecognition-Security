import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet50"):
        super(SiameseNetwork, self).__init__()
        
        # Load Pretrained Backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif backbone == "resnet101":
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Unsupported backbone")
        
        # Remove Fully Connected Layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Embedding Layer
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)  # L2 Normalization
    def get_embedding(self, x):
        with torch.no_grad():
            x = self.backbone(x)
            x = torch.flatten(x, start_dim=1)
            x = self.embedding(x)
            x = F.normalize(x, p=2, dim=1) 
        return x

    
  