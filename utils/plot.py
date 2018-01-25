import numpy as np
import matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt

from utils.parsing import poly_to_mask
from segmentation import get_i_contour
from segmentation import get_threshold_intensity


def plot_segmentation(image, o_mask, i_mask, mode="intensity", algorithm="gmm"):
    i_contour = get_i_contour(image, o_mask, mode=mode, algorithm=algorithm)
    pred_i_mask = poly_to_mask(i_contour, image.shape[0], image.shape[1])

    plt.figure(figsize=(20,20))
    plt.gray()
    plt.subplot(131)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(image + 200*pred_i_mask)
    plt.title("Predicted i_contour")
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(image + 200*i_mask)
    plt.title("True i_contour")
    plt.axis('off')
    plt.show()
    plt.close()


def plot_generator(generator, nb_samples=3):
    count = 0
    plt.figure(figsize=(15,15))
    plt.gray()
    while count < nb_samples:
        X, y = next(generator)
        image = X[0]
        i_mask = y[0][0]
        o_mask = y[1][0]
        plt.subplot(nb_samples, 4, count*4 + 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(nb_samples, 4, count*4 + 2)
        plt.imshow(i_mask)
        plt.title("Inner contour")
        plt.axis('off')
        plt.subplot(nb_samples, 4, count*4 + 3)
        plt.imshow(o_mask)
        plt.title("Outer contour")
        plt.axis('off')
        plt.subplot(nb_samples, 4, count*4 + 4)
        plt.imshow(image + 200*i_mask + 200*o_mask)
        plt.title("Superposition")
        plt.axis('off')
        count += 1
    plt.savefig("output/generator")
    plt.show()
    plt.close()

    
def plot_histogram(image, o_mask, algorithm):

    threshold = get_threshold_intensity(image, o_mask, algorithm=algorithm)
    data = list(image[o_mask])
    plt.hist(data, bins=40, histtype='stepfilled')
    plt.axvline(x=threshold, c='black')
    plt.xlim(0,255)
    plt.title("Normalized pixel intensity")
    plt.savefig("output/histogram")
    plt.show()
    plt.close()
    
