import tensorflow as tf

import sys

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


def parse_function(image_file, label_file):
    """Global parsing function
    :param image_file: Path to the image file
            label_file: Path to the label file
    :return image: np.array of the image
            mask: np.array of the mask
    """
    
    image_file = image_file.decode(sys.getdefaultencoding())
    label_file = label_file.decode(sys.getdefaultencoding())
    dcm_dict = parse_dicom_file(image_file)
    image = dcm_dict["pixel_data"]
    coords_lst = parse_contour_file(label_file)
    mask = poly_to_mask(polygon=coords_lst, width=image.shape[0], height=image.shape[1])
    image = image.astype(np.float32)
    mask = mask.astype(np.float32)
    
    return image, mask

    
def get_files():
    """Get the files where the images and labels are stored
    :return image_files: List of image files
            label_files: List of corresponding label files
    """
    
    image_files = []
    label_files = []
    patient_ids, original_ids = get_corresponding_ids("%s/link.csv"%data_path)
    
    for patient_id, original_id in zip(patient_ids, original_ids):
        nb_slices = len(os.listdir("%s/dicoms/%s"%(data_path,patient_id)))
        i_contour_files = os.listdir("%s/contourfiles/%s/i-contours"%(data_path, original_id))
        for slice_id in range(1,nb_slices+1):
            for contour_file in i_contour_files:
                if int(contour_file[8:12]) == slice_id:
                    image_file = "%s/dicoms/%s/%d.dcm"%(data_path, patient_id, slice_id)
                    label_file = "%s/contourfiles/%s/i-contours/%s"%(data_path, original_id, contour_file)
                    image_files.append(image_file)
                    label_files.append(label_file)
    return image_files, label_files


def get_dataset(config, image_files, label_files):
    """Generate a tf.data.Dataset object
    :param config: Configuration parameters (batch_size, epochs..)
           image_files: List of image files
           label_files: List of corresponding label files
    :return dataset: tf.data.Dataset object
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_files, label_files))
    dataset = dataset.map(lambda image_file, label_file: tf.py_func(parse_function, [image_file, label_file],
                                                                    [tf.float32, tf.float32]))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.batch(config.batch_size)
    return dataset

    
def get_iterator(dataset):
    """Builds an Iterator object corresponding to a tf.data.Iterator object
    :param dataset: tf.data.Dataset object
    :return iterator: tf.data.Iterator object to feed into the neural network
    """
    
    iterator = tf.data.Iterator.from_structure(output_types=dataset.output_types, output_shapes=dataset.output_shapes)
    return iterator


