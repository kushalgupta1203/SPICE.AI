import torch
import torch.nn as nn
import torchvision.models as models

class SolarPanelClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(SolarPanelClassifier, self).__init__()
        
        # Load pre-trained ResNet50 model
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Freeze early layers to prevent overfitting (can be unfrozen later if needed)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Modify the final layer for classification
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout to prevent overfitting
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)
