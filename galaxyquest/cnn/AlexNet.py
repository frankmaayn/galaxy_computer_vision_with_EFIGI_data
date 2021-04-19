import torch.nn as nn
import torch

def alexnet_model(num_classes):
    alexnet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    # Make changes to the layers to work with the EFIGI dataset.
    alexnet_model.classifier[4] = nn.Linear(4096, 1024)
    alexnet_model.classifier[6] = nn.Linear(1024, num_classes)
    return alexnet_model