# DICOM_MRI


### Code Architecture

The main file is called main.py. Executing this file corresponds to part 1 and part 2.


## Part 1: Parse the DICOM images and Contour Files

##### Task: Using the functions given above, build a pipeline that will parse the DICOM images and i-contour contour files, making sure to pair the correct DICOM image(s) with the correct contour file. After parsing each i-contour file, make sure to translate the parsed contour to a boolean mask.

In part 1 of the main.py file, there are 2 function calls:

-**get_data()** (from utils.preprocess): This function collects all the images and the corresponding labels and generate numpy arrays. It builds on top of the functions from the starter code parsing.py that were not modified:

* The first step: **get\_corresponding\_ids()** is to link the DICOM files with the contour files using the link file 

* To map all DICOM images to their corresponding contours, I am using the reference within the contour filenames. The DICOM image files are numbered by the succesive slides in the MRI scan: "48.dcm" corresponds to the 48th slide. For instance, "IM-0001-**0048**-icontour-manual.txt" corresponds to "48.dcm". This is implemented in **get\_data()**.

-**plot_segmentation()** (from utils.plot): This function is used as a verification step (cf Q1). It samples **nbsamples** examples from the dataset and plots the DICOM image with the corresponding segmentation label.

#### Q1: How did you verify that you are parsing the contours correctly?

I plotted some segmentation samples to verify that I extracted the correct labels. We can see on the following images that the segmentation makes sense.

![Segmentation example]
(output/segmentation_samples.png)


#### Q2: What changes did you make to the code, if any, in order to integrate it into our production code base?
 
I looped over all patients, all DICOM files and all contour files to identify the correponding pairs between images and targets and save them into np.arrays: "images" and "labels".


#### Ways of improvements:
The mapping between the 2D DICOM images and the contour files could be improved. The contours are given only for a subset of the images (I believe because the label is very expensive). My approach was to only retain 2D images with a corresponding label and to discard images that don’t have a label (for instance contour “IM-0001-0048-icontour-manual.txt” corresponds to image “48.dcm”, etc…).

However with this approach, I am wasting 90% of the images because there are roughly only 20 contours for 200 slide images. I was thinking that I could generate the missing labels by K-Nearest neighbors or linear interpolation. If we assume that the semantic segmentation varies continuously from slide to slide within the 3D image, it would be reasonable to input the missing segmentation labels using the segmentation labels of neighbouring slides. Using this method, I could have more training examples but the labels would be more noisy. I didn't implement this last step but it would fit within the **get\_data()** function.


## Part 2: Model training pipeline
 

Using the saved information from the DICOM images and contour files, add an additional step to the pipeline that will load batches of data for input into a 2D deep learning model. This pipeline should meet the following criteria:
 
* Cycles over the entire dataset, loading a single batch (e.g. 8 observations) of inputs (DICOM image data) and targets
(boolean masks) at each training step.
 
* A single batch of data consists of one numpy array for images and one numpy array for targets.

* Within each epoch (e.g. iteration over all studies once), samples from a batch should be loaded randomly from the
entire dataset.
 
The ouptut of part 1 are 2 np.arrays of "images" and their corresponding "labels" for the full dataset.

The aim of part 2 is to randomly sample **batchsize** samples without replacement from the dataset until there are no more samples (end of the epoch) and repeat the procedure for each epoch. To do so I could use functions from numpy like np.random.sample() or np.random.permutation(). However, I decided to use an API from tensorflow called tf.data that automates the generation of training batches to feed into a neural network.

* The first step is to create a Config object (config.py) that holds all the parameters: **batch\_size**, **epochs**, and later all the learning parameters to train the model.

* The second step is to create a **Model** object (model.py) that is initialized using the **config** object, the **images** and **labels** np.arrays. This **Model** object has an attribute called **dataset\_op**, which is a tf.data.Iterator object initialized using a tf.data.Dataset object (see utils.dataset.py). The benefit of this method is that the shuffling and batch creation is done within the tf.data.Dataset object creation and the tf.data.Iterator object yields tf.tensors **image** and **label** of desired shape (**batch\_size**) at every training step using **Iterator.get_next()**. These tensors can be directly used within a tensorflow model in the next step of the challenge.

For the purpose of this exercise, I added a method to the Model object called **get\_train\_batches()** to run a tensorflow session and evaluate the **image** and **label** tensors at every training steps. The resulting np.arrays are stored as tuples within a python list.


##### Q1: Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2? If so, what? If not, is there anything that you can imagine changing in the future?

I didn't change anything from the Part 1 because I use the np.array generated in Part 1 to create a tf.data.Dataset object. In the future, I will need to split the dataset into a training, validation and test set (or some cross-validation). I will also normalize the datasets.

In the future, if the dataset is too big I won't be able to fit the full np.array in memory. In this case, I would only generate the filenames of the images and labels and parse the images "on the go" at every training step.

##### Q2: How do you/did you verify that the pipeline was working correctly?
 
I created a method called **get\_train\_batches()** in the model object to generate a list of numpy arrays of images and labels for each **epochs**. I checked that these arrays were both of shape **batchsizeX256X256**.

I also used a "fake" label corresponding to the image index in the original dataset to verify that the batches were samples randomly and without replacement in every epochs.
 
##### Q3: Given the pipeline you have built, can you see any deficiencies that you would change if you had more time? If not, can you think of any improvements/enhancements to the pipeline that you could build in?

In the future, if the dataset is too big I won't be able to fit the full np.array in memory. In this case, I would only generate the filenames of the images and labels and parse the images "on the go" at every training step.

In Part 1, I was talking about the value inputation to avoid wasting 90% of the data, by inputing the missing labels using neighboring slides in the image.

I would also add some preprocessing steps to the Dataset object: normalization, train/val/test split, cross validation, etc...
 