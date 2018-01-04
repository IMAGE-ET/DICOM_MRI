import tensorflow as tf

class Dataset:
    
    def __init__(self, config):
        """Builds a Dataset object corresponding to a tf.data.Dataset object
        :param config: Configuration parameters
        """
        
        self.config = config
        
    def get_dataset(self, images, labels):
        """Generate a tf.data.Dataset object
        :param images: np.array of the DICOM images
               labels: np.array of the boolean masks
        :return dataset: tf.data.Dataset object
        """
        
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.batch(self.config.batch_size)
        return dataset
    
def get_iterator(dataset):
    """Builds an Iterator object corresponding to a tf.data.Iterator object
    :param dataset: tf.data.Dataset object
    """
    
    iterator = tf.data.Iterator.from_structure(output_types=dataset.output_types, output_shapes=dataset.output_shapes)
    return iterator
