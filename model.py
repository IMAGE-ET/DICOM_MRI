import numpy as np

from utils.data import get_data
from utils.generator import Generator

class Model:
    """
    Class to define the model. For the moment, only the dataset loading part is implemented
    """
    def __init__(self, config):
        """
        :param config: Configuration parameters (batch_size, epochs..)
        """
        
        self.config = config


    def add_data(self):
        """Create a dataset_op which corresponds to a tf.data.Iterator
        """
        partition, labels = get_data()
        generator = Generator(self.config)
        self.train_generator = generator.generate(partition['train'], labels)
        self.val_generator = generator.generate(partition['val'], labels)
        
        
