import pandas as pd
import cv2
import numpy as np
import os

class GBMDataset(object):
    def __init__(self, image_root_path, label_file_path):
        """
            image_root_path: folder containing EFIGI images.
            label_file_path: path to the file containing labels (EFIGI_labels.csv)
        """
        self.image_root_path = image_root_path

        # Load the label file as a dataframe.
        # Take out the labels we need here.
    
        label_dataframe = pd.read_csv(label_file_path)
        self.pgc_ids = label_dataframe["PGCname"]

    def __len__(self):
        return len(self.pgc_ids)

    def __getitem__(self, idx):

        # Return a dictionary like such:
        # { 'pgc_id': <pgc_id>, 'image_features': <list of image features>}

        pgc_id = self.pgc_ids[idx]

        current_image = cv2.imread(os.path.join(self.image_root_path, pgc_id + ".png"))
        current_image_features = self.preprocess_image(current_image)

        return {
            'pgc_id': pgc_id,
            'features': current_image_features
        }
    
    def split_dataset(self):
        # Thinking of adding logic here to split the data into 3 sets:
        # train, validate, test.
        

