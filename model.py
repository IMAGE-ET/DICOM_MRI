import tensorflow as tf
import numpy as np

from utils.dataset import Dataset
from utils.dataset import get_iterator

class Model:
    """
    Class to define the model. For the moment, only the dataset loading part is implemented
    """
    def __init__(self, config, images, labels):
        """
        :param config: Configuration parameters
               images: np.array of the DICOM images
               labels: np.array of the boolean masks
        """
        
        self.config = config
        self.len_trainset = len(images)
        self.add_dataset_op(images, labels)
        #self.add_placeholders()
        #self.add_model()
        #self.add_pred_op()
        #self.add_loss_op()
        #self.add_train_op()

    def add_dataset_op(self, images, labels):
        """Create a dataset_op which corresponds to a tf.data.Iterator
        :param images: np.array of the DICOM images
               labels: np.array of the boolean masks
        """
        
        dataset = Dataset(self.config)
        dataset = dataset.get_dataset(images, labels)
        self.iterator = get_iterator(dataset)
        self.image, self.label = self.iterator.get_next()
        self.dataset_op = self.iterator.make_initializer(dataset)
    
    def get_train_batches(self):
        """Create a list of np.array batches for every training steps 
        :return batches: List of batches.
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
    