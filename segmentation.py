#import cv2
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from skimage import measure
from scipy import ndimage as ndi



def get_i_contour(image, o_mask, threshold):
    
    pred_i_mask = (image*o_mask>threshold).astype("bool")
    pred_i_mask = ndi.binary_dilation(pred_i_mask, iterations=2)
    pred_i_mask = ndi.binary_fill_holes(pred_i_mask)
    contour = measure.find_contours(pred_i_mask, 0.5, fully_connected='low', positive_orientation='low')[0]
    contour = [(y,x) for x,y in contour]
    
    return contour


def get_threshold_intensity(image, o_mask, algorithm="gmm"):
    
    assert algorithm in ["gmm", "kmeans"], "clustering algorithm not available. Please choose between \"gmm\" and \"kmeans\""
    
    data = list(image[o_mask])
    data.sort()
    data = np.array(data).reshape((-1,1))
    if algorithm == "gmm":
        model = GaussianMixture(n_components=2)
    if algorithm == "kmeans":
        model = KMeans(n_clusters=2)
    
    model.fit(data)
    pred = model.predict(data)
    threshold = data[np.argmax(pred!=pred[0])][0]
    
    return threshold


