from config import Config
from utils.data import get_data
from utils.generator import Generator
from utils.plot import plot_generator
from utils.plot import plot_histogram
from utils.plot import plot_segmentation
from segmentation import get_i_contour


n_samples = 50

if __name__ == "__main__":
    #Part 1: Parse the o-contours
    print ("\nPart 1 :\n")

    #Extract the data: images and segmentation (inner contours and outer contours)
    config = Config()
    image_files, labels = get_data()
    generator = Generator(config).generate(image_files, labels)
    #Plot some samples
    plot_generator(generator, n_samples)


    #Part 2: Heuristic LV segmentation
    print ("\nPart 2 :\n")

    #Extract samples from the generator
    images, (i_masks, o_masks) = next(generator)

    #Q2.1.: Intensity-based prediction
    print ("Intensity-based prediction")
    for image, o_mask, i_mask in zip(images[0:n_samples], o_masks[0:n_samples], i_masks[0:n_samples]):
        #Extract the contour using the intensity-based method
        i_contour = get_i_contour(image, o_mask, mode="intensity", algorithm="gmm")
        plot_histogram(image, o_mask, algorithm="gmm")
        #Plot the predicted segmentation
        plot_segmentation(image, o_mask, i_mask, mode="intensity", algorithm="gmm")

    #Q2.2.: Edges-based prediction
    print ("Edges-based prediction")
    for image, o_mask, i_mask in zip(images[0:n_samples], o_masks[0:n_samples], i_masks[0:n_samples]):
        #Extract the contour using the edge-based method
        i_contour = get_i_contour(image, o_mask, mode="edges")
        #Plot the predicted segmentation
        plot_segmentation(image, o_mask, i_mask, mode="edges")
        
        
        
