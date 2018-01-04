import os
import numpy as np
import pandas as pd
from utils.parsing import parse_contour_file, parse_dicom_file, poly_to_mask

data_path = "final_data"

def get_corresponding_ids(filepath):
    """Extract the corresponding patient_ids and original_ids
    :param filepath: Path to the link file
    :return patient_ids: List of patient_ids
            original_ids: List of original_ids
    """
    
    link = pd.read_csv(filepath)
    patient_ids = link.loc[:,"patient_id"]
    original_ids = link.loc[:,"original_id"]
    return patient_ids, original_ids

def get_image(patient_id, slice_id):
    """Parse the image associated with patient_id and slice_id
    :param patient_id: Name of the patient folder
           slice_id: Identifier of the DICOM file
    :return image: np.array representing the image
    """
    
    dcm_file = "%d.dcm"%slice_id
    dcm_dict = parse_dicom_file("%s/dicoms/%s/%s"%(data_path, patient_id, dcm_file))
    image = dcm_dict["pixel_data"]
    return image

def get_nb_slices(patient_id):
    """Compute the nb of slices in the full 3D image
    :param patient_id: Name of the patient folder
    :return nb_slices: Nb of slices in the full 3D image
    """
    
    nb_slices = len(os.listdir("%s/dicoms/%s"%(data_path,patient_id)))
    return nb_slices

def get_contour_files(original_id, contour_type="i"):
    """Parse the contour file associated with original_id and contour_type
    :param original_id: Name of the contour folder
           contour_type: "i" or "o" depending on the contour
    :return contour_files: Filenames of the contours
    """
    
    contour_files = os.listdir("%s/contourfiles/%s/%s-contours"%(data_path, original_id, contour_type))
    return contour_files    
    
def get_data(interpolation=False):
    """Get the data
    :param interpolation: Whether to perform value imputation for missing labels (not implemented).
    :return inputs: Numpy array of all 2D images
            labels: Numpy array of all segmentations
    """
    
    images = []
    labels = []
    patient_ids, original_ids = get_corresponding_ids("%s/link.csv"%data_path)
    
    for patient_id, original_id in zip(patient_ids, original_ids):
        nb_slices = get_nb_slices(patient_id)
        i_contour_files = get_contour_files(original_id, contour_type="i")
        for slice_id in range(1,nb_slices+1):
            if interpolation == False:
                for contour_file in i_contour_files:
                    if int(contour_file[8:12]) == slice_id:
                        coords_lst = parse_contour_file("%s/contourfiles/%s/i-contours/%s"%(data_path, original_id, contour_file))
                        image = get_image(patient_id, slice_id)
                        mask = poly_to_mask(polygon=coords_lst, width=image.shape[0], height=image.shape[1])
                        images.append(image)
                        labels.append(mask)
            if interpolation == True:
                pass
    images = np.asarray(images)
    labels = np.asarray(labels)

    return images, labels


