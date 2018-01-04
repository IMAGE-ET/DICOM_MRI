import numpy as np
import matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt

def plot_segmentation(images, labels, nb_samples=3):
    """Plot a selection of images and their segmentation
    :param images: np.array of the DICOM images
           labels: np.array of the boolean masks
           nb_samples: Nb of samples to plot
    """
    
    fig = plt.figure(figsize=(20,6*nb_samples))
    count = 0
    for idx in np.random.randint(0,len(images),nb_samples):
        image = images[idx]
        label = labels[idx]
        plt.subplot(nb_samples, 3, count*3 + 1)
        plt.imshow(image)
        plt.title("Raw image")
        plt.subplot(nb_samples, 3, count*3 + 2)
        plt.imshow(label)
        plt.title("Raw Segmentation")
        plt.subplot(nb_samples, 3, count*3 + 3)
        plt.imshow(image+label*np.max(image))
        plt.title("Segmentation on the image")
        count += 1
    print ("Segmentation samples saved to output/")
    fig.savefig('output/segmentation_samples.png')

    