from config import Config
from model import Model
from utils.data_Q1 import get_data
from utils.plot import plot_segmentation


if __name__ == "__main__":
    
    print ("Part 1 :\n")
    #Get the images and corresponding labels
    images, labels = get_data()
    #Plot 3 samples to check the segmentation
    plot_segmentation(images, labels, nb_samples=3)
    #Shape verification
    print ("Size of the image dataset:", images.shape)
    print ("Size of the label dataset:", labels.shape)
   
    print ("\nPart 2 :\n")
    #Load the parameters
    config = Config()
    #Build a model
    model = Model(config)
    #Create a list of np.arrays for each batch
    batches = model.get_train_batches()
    #Shape verification
    print ("Size of the image batch:", batches[0][0].shape)
    print ("Size of the label batch:", batches[0][1].shape)