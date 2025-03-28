import torch
import torch.nn as nn
import torchvision.models as models

class SolarPanelClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(SolarPanelClassifier, self).__init__()

        # Load pre-trained ResNet50 model
        self.model = models.resnet50(weights="IMAGENET1K_V1")  # Ensure correct weight loading

        # Unfreeze the last few layers for fine-tuning
        for name, param in self.model.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:  # Unfreeze last 2 blocks
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Add Adaptive Pooling for flexibility
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Modify the final layer for classification
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),  # Added BatchNorm for stability
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout to prevent overfitting
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)
