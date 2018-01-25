#import cv2
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from skimage import measure
from skimage import feature
from scipy import ndimage as ndi



def get_i_contour(image, o_mask, mode="intensity", algorithm="gmm"):
    """Compute the predicted inner contour (blood pool) using the outer contour
    Source: http://scikit-image.org/docs/0.12.x/auto_examples/xx_applications/plot_coins_segmentation.html
    
    :param image (np.array): Image to plot
    :param o_mask (no.array): Segmentation mask of the outer contour
    :param mode (str, optional): Mode of segmentation: "intensity" or "edges"
    :param algorithm (str, optional): Algorithm for thresholding: "gmm" or "kmeans"
    :return: list of tuples holding x, y coordinates of the inner contour
    """
    
    #Compute the predicted inner contour mask
    if mode == "intensity":
        threshold = get_threshold_intensity(image, o_mask, algorithm=algorithm)
        pred_i_mask = (image*o_mask>threshold).astype("bool")
    if mode == "edges":
        pred_i_mask = feature.canny(image, sigma=3, low_threshold=10, high_threshold=25, mask=o_mask, use_quantiles=False)
    
    #Post-processing: Mathematical morphology operations
    pred_i_mask = ndi.binary_dilation(pred_i_mask, iterations=2)
    pred_i_mask = ndi.binary_fill_holes(pred_i_mask)
    
    #Extract the contour from the predicted inner contour mask
    contour = measure.find_contours(pred_i_mask, 0.5)[0]
    contour = [(y,x) for x,y in contour]

    return contour


def get_threshold_intensity(image, o_mask, algorithm="gmm"):
    """Compute the threshold to separate blood pool pixels from muscle pixels
    Source: http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MORSE/threshold.pdf

    :param image (np.array): Image to plot
    :param o_mask (no.array): Segmentation mask of the outer contour
    :param algorithm (str, optional): Algorithm for thresholding: "gmm" or "kmeans"
    :return: Scalar classification threshold between blood pool pixels and muscle pixels
    """
    
    assert algorithm in ["gmm", "kmeans"], "clustering algorithm not available. Please choose between \"gmm\" and \"kmeans\""
    
    #Extract the pixel intensity values within the outer contour
    data = list(image[o_mask])
    data.sort()
    data = np.array(data).reshape((-1,1))
    
    #Build a model to cluster the data
    if algorithm == "gmm":
        model = GaussianMixture(n_components=2)
    if algorithm == "kmeans":
        model = KMeans(n_clusters=2)   
    model.fit(data)
    pred = model.predict(data)
    
    #Compute the threshold based on the clustering of pixels
    threshold = data[np.argmax(pred!=pred[0])][0]
    
    return threshold


