import torch
import torch.nn as nn
import torchvision.models as models

class SolarPanelClassifier(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(SolarPanelClassifier, self).__init__()
        
        # Load a pre-trained ResNet50 model
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Modify the final layer for multi-label classification
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),  # Dropout for regularization
            nn.Linear(in_features, num_classes)  # Multi-label output
        )

    def forward(self, x):
        return self.base_model(x)
