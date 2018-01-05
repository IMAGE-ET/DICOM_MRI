import tensorflow as tf
import numpy as np

from utils.data_Q2 import get_dataset
from utils.data_Q2 import get_iterator
from utils.data_Q2 import get_files

class Model:
    """
    Class to define the model. For the moment, only the dataset loading part is implemented
    """
    def __init__(self, config):
        """
        :param config: Configuration parameters (batch_size, epochs..)
        """
        
        self.config = config
        self.add_dataset_op()

    def add_dataset_op(self):
        """Create a dataset_op which corresponds to a tf.data.Iterator
        """
        #Create the tf.data.Dataset object
        image_files, label_files = get_files()
        self.len_trainset = len(image_files)
        dataset = get_dataset(self.config, image_files, label_files)
        
        #Create the tf.data.Iterator object
        self.iterator = get_iterator(dataset)
        self.image, self.label = self.iterator.get_next()
        self.dataset_op = self.iterator.make_initializer(dataset)
    
    def get_train_batches(self):
        """Create a list of np.array batches for every training steps 
        :return batches: List of batches (np.arrays)
        """
        
        batches = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.config.epochs):
                sess.run(self.dataset_op)
                nb_batch = int(np.ceil(self.len_trainset/self.config.batch_size))
                for batch in range(nb_batch):
                    image_array, label_array = sess.run([self.image, self.label])
                    batches.append((image_array, label_array))
        return batches
    