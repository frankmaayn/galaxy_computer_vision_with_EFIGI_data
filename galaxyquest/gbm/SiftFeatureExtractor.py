import cv2
import os
import numpy as np
import pandas as pd

# Utility script that takes as input a file formatted like EFIGI_labels.csv,
# a dataset of images,
# and extracts features using SIFT.

def extract_image_features(image_root_path, label_dataset_path,  output_file: str = None):
    # Dataframe of image labels.
    # Also contains PGC ID's of all images.
    label_dataframe = pd.read_csv(label_dataset_path)
    pgc_ids = np.array(label_dataframe["PGCname"])

    # Load images into a list.
    images = load_images_from_folder(pgc_ids, image_root_path)

    dataframe_np = np.reshape(pgc_ids, (-1, 1))
    column_names = ["pgc_id"]
    features = []

    for image in images:
        # The loop that extracts features.        
        # Get 2 keypoints.
        sift = cv2.SIFT_create(2)

        # Get SIFT histograms for them,
        # and append these to the feature list.
        kp, descriptions = sift.detectAndCompute(image, None)
        descriptions = [n for description in descriptions[0:2] for n in description]
        features.append(descriptions)
    
    features = np.array(features)
    
    for i in range(features.shape[1]):
        column_names.append("SIFT_" + str(i))

    dataframe_np = np.append(dataframe_np, features, 1)

    feature_dataframe = pd.DataFrame(
        dataframe_np,
        columns = column_names
    )

    feature_dataframe.to_csv(output_file)


def load_images_from_folder(pgc_ids, image_root_path):
    images = []
    for pgc_id in pgc_ids:
        image = cv2.imread(os.path.join(image_root_path, pgc_id + ".png"))
        if image is not None:
            images.append(image)
    return images

if __name__ == "__main__":
    extract_image_features("X:\efigi\efigi_png_gri-1.6\efigi-1.6\png",  "../../datasets/EFIGI_labels.csv", "EFIGI_features.csv")