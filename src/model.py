import torch
import torch.nn as nn
import torch.nn.functional as F

class DefectClassificationModel(nn.Module):
    def __init__(self, num_classes=6):
        super(DefectClassificationModel, self).__init__()
        # Dynamic feature extractor for surface inspection textures
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 224 -> 112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 112 -> 56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 56 -> 28
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # Quick sanity shape check
    model = DefectClassificationModel()
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"✅ Model structural check passed. Output matrix shape: {output.shape}")