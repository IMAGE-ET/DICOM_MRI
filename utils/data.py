import numpy as np
import pandas as pd
import os

data_path = "final_data"

def get_data():
    """Get the filenames for the input images and their corresponding contour filenames
    
    :return image_files: List of image filenames to process
    :return labels: Dictionary of labels (contour filenames for every image)
    """
    
    image_files = []
    labels = {}
    patient_ids, original_ids = get_corresponding_ids()
    for patient_id, original_id in zip(patient_ids, original_ids):
        #Extract all the files
        images = os.listdir("%s/dicoms/%s"%(data_path,patient_id))
        i_contour_files = os.listdir("%s/contourfiles/%s/i-contours"%(data_path, original_id))
        o_contour_files = os.listdir("%s/contourfiles/%s/o-contours"%(data_path, original_id))
        #Filter the files that have both i-contour and o-contour labels
        for image in images:
            for i_contour_file in i_contour_files:
                for o_contour_file in o_contour_files:
                    if int(image.split('.')[0]) == int(i_contour_file[8:12]) == int(o_contour_file[8:12]):
                        image_file = "%s/dicoms/%s/%s"%(data_path, patient_id, image)
                        image_files.append(image_file)
                        labels[image_file] = ["%s/contourfiles/%s/i-contours/%s"%(data_path, original_id, i_contour_file),
                                              "%s/contourfiles/%s/o-contours/%s"%(data_path, original_id, o_contour_file)]
    
    return image_files, labels


def get_corresponding_ids():
    """Extract the corresponding patient_ids and original_ids
    
    :return patient_ids: List of patient_ids
    :return original_ids: List of original_ids
    """
    
    link = pd.read_csv("%s/link.csv"%data_path)
    patient_ids = link.loc[:,"patient_id"]
    original_ids = link.loc[:,"original_id"]
    
    return patient_ids, original_ids




