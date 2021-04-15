import cv2 as cv
import numpy as np
import pandas as pd
import os

if __name__ == "__main__":
    convert_image_to_binary_threshold("C:\Users\serde\Desktop\COMP 442\Project\galaxy_computer_vision_with_EFIGI_data\images",  "../../galaxy_computer_vision_with_EFIGI_data/datasets/EFIGI_labels.csv")


# Loads the images
def load_images_from_folder(pgc_ids, image_root_path):
    images = []
    for pgc_id in pgc_ids:
        image = cv.imread(os.path.join(image_root_path, pgc_id + ".png"))
        if image is not None:
            images.append(image)
    return images

#Converts the images using binary thresholding
def convert_image_to_binary_threshold(image_root_path, label_dataset_path):

    #File path for the new images
    new_image_root_path = '../../galaxy_computer_vision_with_EFIGI_data/threshold_image/'

    #Gets the dataset that contains the image names
    label_dataframe = pd.read_csv(label_dataset_path)

    #Converts it to an array
    pgc_ids = np.array(label_dataframe["PGCname"])
    
    # Load images into a list.
    images = load_images_from_folder(pgc_ids, image_root_path)
   
    i = 0;
    for image in images:

        #Use binary thresholding
        ret,th1 = cv.threshold(image,127,255,cv.THRESH_BINARY)

        #Save the image to the new folder
        cv.imwrite(os.path.join(new_image_root_path,pg_ids[i] + '.png'),th1)
        i = i +1;





