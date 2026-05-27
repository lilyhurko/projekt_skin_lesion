import torch
import torch.nn as nn
from torchvision import models

class SkinEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SkinEfficientNet, self).__init__()

        self.network = models.efficientnet_b0(weights='DEFAULT')
        
        in_features = self.network.classifier[1].in_features
        

        self.network.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), 
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def get_model_for_level(level):
    return SkinEfficientNet(num_classes=2)