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
    column_names = ["pgc_id"]
    features = []
    empty_kp_counter = 0
    pgc_ids_filtered = []


    for i in range(len(images)):
        # The loop that extracts features.        
        # Get 2 keypoints.
        sift = cv2.SIFT_create(2)

        # Get SIFT histograms for them,
        # and append these to the feature list.
        kp, descriptions = sift.detectAndCompute(images[i], None)

        if descriptions is not None:
            if len(descriptions) >= 2:
                description_list = []
                for description in descriptions[0:2]:
                    for n in description:
                        description_list.append(n)
                features.append(description_list)
                pgc_ids_filtered.append(pgc_ids[i])
                #descriptions = [n for description in descriptions[0:2] for n in description]
        else:
            empty_kp_counter += 1

    print(empty_kp_counter)
    features = np.array(features)

    
    for i in range(features.shape[1]):
        column_names.append("SIFT_" + str(i))

    dataframe_np = np.reshape(pgc_ids_filtered, (-1, 1))
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
    extract_image_features(r"C:\Users\dpale\Desktop\Projects\galaxy_computer_vision_with_EFIGI_data\threshold_image",  r"C:\Users\dpale\Desktop\Projects\galaxy_computer_vision_with_EFIGI_data\datasets\EFIGI_labels.csv", r"EFIGI_SIFT_feature_data_2kp_threshold.csv")