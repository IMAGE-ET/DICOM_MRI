from config import Config
from model import Model
from utils.preprocess import get_data
from utils.plot import plot_segmentation


if __name__ == "__main__":
    
    # Part 1:
    #Get the images and corresponding labels
    images, labels = get_data()
    #Plot 3 samples to check the segmentation
    plot_segmentation(images, labels, nb_samples=3)
    
    # Part 2:
    #Load the parameters
    config = Config()
    #Build a model (only the data loading is implemented: shuffling+batch)
    model = Model(config, images, labels)
    #Creates a list of np.array for each batch
    batches = model.get_train_batches()

    #Shape verification
    for batch in batches:
        print(batch[0].shape)
        print(batch[1].shape)