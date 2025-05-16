import torch.nn as nn
from torchvision import models

def get_finetuned_resnet(num_classes=101, pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
