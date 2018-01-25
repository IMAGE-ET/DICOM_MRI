import numpy as np

from utils.parsing import parse_contour_file
from utils.parsing import parse_dicom_file
from utils.parsing import poly_to_mask


class Generator(object):
    """Generator object to generate batches of samples
    Source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    """
    
    def __init__(self, config):
        
        self.config = config
    
    def generate(self, image_files, labels):
        """Generates batches of samples
        
        :param image_files: List of image filenames to process
        :param labels: Dictionary of labels (contour filenames for every image)
        :yield X: Batch of image data
        :yield y: Batch of segmentation data (inner contour and outer contour)
        """
        
        while 1:
            indexes = self.__get_exploration_order(image_files)
            imax = int(len(indexes)/self.config.batch_size)
            for i in range(imax):
                image_files_batch = [image_files[k] for k in indexes[i*self.config.batch_size:(i+1)*self.config.batch_size]]
                X, y = self.__data_generation(image_files_batch, labels)

                yield X, y


    def __get_exploration_order(self, image_files):
        """Generates order of exploration
        
        :param image_files: List of image filenames to process
        :return indexes: Shuffled ordering of the image filenames.
        """
        
        indexes = np.arange(len(image_files))
        if self.config.shuffle == True:
            np.random.shuffle(indexes)

        return indexes


    def __data_generation(self, image_files_batch, labels):
        """Generates data of batch_size samples
        
        :param image_files_batch: List of image filenames in the batch
        :param labels: Dictionary of labels (contour filenames for every image)
        :return X: Batch of image data
        :return y: Batch of segmentation data (inner contour and outer contour)
        """
        
        X = []
        i_y = []
        o_y = []
        for image_file in image_files_batch:
            image, i_mask, o_mask = parse_function(image_file, labels)
            X.append(image)
            i_y.append(i_mask)
            o_y.append(o_mask)
        X = np.asarray(X)
        i_y = np.asarray(i_y)
        o_y = np.asarray(o_y)
        y = [i_y, o_y]
        
        return X, y

    
def parse_function(image_file, labels):
    """Wrapper around the parsing functions in utils.parsing
        
    :param image_file: filepath to the DICOM file to parse
    :param labels: Dictionary of labels (contour filenames for every image)
    :return image: Image to plot
    :return i_mask: Segmentation mask of the outer contour
    :return o_mask: Segmentation mask of the inner contour
    """

    #Parse and normalize the image
    dcm_dict = parse_dicom_file(image_file)
    image = dcm_dict["pixel_data"]
    image = image.astype("float")
    image *= 255.0/image.max()
    
    #Parse the inner contour
    i_contour_file = labels[image_file][0]
    i_coords_lst = parse_contour_file(i_contour_file)
    i_mask = poly_to_mask(polygon=i_coords_lst, width=image.shape[0], height=image.shape[1])
    
    #Parse the outer contour
    o_contour_file = labels[image_file][1]
    o_coords_lst = parse_contour_file(o_contour_file)
    o_mask = poly_to_mask(polygon=o_coords_lst, width=image.shape[0], height=image.shape[1])
    
    return image, i_mask, o_mask
    
    
    