import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
import numpy as np
import torch

class EFIGIDataset(Dataset):
    def __init__(self, image_root, label_path):
        self.image_root = image_root
        self.label_dataframe = pd.read_csv(label_path)
        self.labels = np.array(self.label_dataframe["category_label"])
        self.labels = torch.from_numpy(self.labels)
        self.pgc_ids = list(self.label_dataframe["PGCname"])

        # Normalize images into [0, 1] with the expected mean and stds.
        self.transform = transforms.Compose([
            transforms.Resize(227),
            transforms.CenterCrop(227),
            transforms.ToTensor(), # normalize to [0, 1]
            transforms.Normalize(
                mean=[0.485],
                std=[0.229],
            ),
        ])
        
    def __len__(self):
        return len(self.label_dataframe)
    
    def __getitem__(self, idx):
        # Get image label and image,
        # apply transform to get it into AlexNet's expected format.
        image_path = os.path.join(self.image_root, self.pgc_ids[idx] + ".png")
        image = Image.open(image_path)
        image = self.transform(image)
        image_label = self.labels[idx]
        pgc_id = self.pgc_ids[idx]
        return {
            "image": image,
            "label": image_label,
            "pgc_id": pgc_id
        }

    def get_labels(self, idx = None):
        if idx is None:
            return list(self.label_dataframe["category_label_names"])
        else: 
            return list(self.label_dataframe.loc[idx,"category_label_names"])
