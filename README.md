# DICOM_MRI


### Code Architecture

The main file is called main.py. Executing this file corresponds to part 1 and part 2.


### Part 1: Parse the DICOM images and Contour Files

##### Task: Using the functions given above, build a pipeline that will parse the DICOM images and i-contour contour files, making sure to pair the correct DICOM image(s) with the correct contour file. After parsing each i-contour file, make sure to translate the parsed contour to a boolean mask.

In part 1 of the main.py file, there are 2 function calls:

-get_data() (from utils.preprocess): This function collects all the images and the corresponding labels and generate numpy arrays. It builds on the functions from the starter code parsing.py that were not modified:

* The first step is to link the DICOM files with the contour files using the link file: *get\_corresponding\_ids()*

* Then, I realized that for one patient, there are roughly 200 DICOM images corresponding to the successive slides in the MRI scans. On the other hand, the contour files are of the form : "IM-0001-0048-icontour-manual.txt" where "0048" in this example corresponds to slide 48. I am therefore using this reference to map all DICOM images to their corresponding contour: *get\_data()*

-plot_segmentation() (from utils.plot): This function samples **nbsamples** examples from the dataset and plot the DICOM image with the corresponding label. 

#### Q1: How did you verify that you are parsing the contours correctly?

I plotted some segmentation samples to verify that I extracted the correct labels:

![Image of Yaktocat]
(/Users/cedoz/DICOM_MRI/output/segmentation_samples.png)


#### Q2: What changes did you make to the code, if any, in order to integrate it into our production code base?
 
I looped over all patients, all DICOM files (image slide) and all contour files to identify the correponding pairs image/target and save them into np.arrays: "images" and "labels".


#### Ways of improvements:
When I am doing the mapping between the 2D DICOM images and the contour files, the contours are present only for a subset of the images (I believe because the label is very expensive). My approach was to only retain 2D images with a corresponding label and to discard images that don’t have a label (for instance contour “IM-0001-0048-icontour-manual.txt” corresponds to image “48.dcm”, etc…).

However with this approach, I am wasting 90% of the images because there are only 20 contours for 200 slide images. I was thinking that I could generate the missing labels by K-Nearest neighbors or linear interpolation. If we assume that the semantic segmentation varies continuously from slide to slide within the 3D image, it would be reasonable to input the missing segmentation labels using the segmentation labels of neighbouring slides. Using this method, I could have more training examples but the labels would be more noisy.


### Part 2: Model training pipeline
 

Using the saved information from the DICOM images and contour files, add an additional step to the pipeline that will load batches of data for input into a 2D deep learning model. This pipeline should meet the following criteria:
 
* Cycles over the entire dataset, loading a single batch (e.g. 8 observations) of inputs (DICOM image data) and targets
(boolean masks) at each training step.
 
* A single batch of data consists of one numpy array for images and one numpy array for targets.

* Within each epoch (e.g. iteration over all studies once), samples from a batch should be loaded randomly from the
entire dataset.
 
The goal of this part is to feed the dataset into a 2D DL model. At the end of part 1, I have a np.array of images and a np.array of corresponding labels.

The goal is to randomly sample **batchsize** samples without replacement from the training set until there are no more samples (end of the epoch) and repeat the procedure **epochs** times. To do that I could use functions like np.random.sample() and np.random.permutation() for the shuffling. However, I decided to use an API from tensorflow called tf.data that automates the generation of training batches to feed into a neural network. 

* The first step is to create a Config object (config.py) that holds all the parameters: batchsize, epochs, and later all the learning parameters to train the model.

* The second step is to create a Model object (model.py) that is initialized using the **config** object, the **images** and **labels** np.arrays. This model objects a tensorflow **dataset\_op**, which is a tf.data.Iterator object initialized using a tf.data.Dataset object (see utils.dataset.py).

The benefit of this method is that the shuffling and batch creation is done within the tf.data.Dataset object creation and the tf.data.Iterator object will yield tf.tensors **image** and **label** of desired shape at every training step using **Iterator.get_next()**. This tensors can be directly used within a tensorflow model in the next step of the challenge.

For the purpose of this exercise, I added a function to the Model object called **get\_train\_batches()** to run a tensorflow session and evaluate the **image** and **label** tensors at every training steps. The resulting np.array are stored as tuples within a python list.

The idea is to first create a tf.data.Dataset object that  I create an Iterator object that would basically be the input to my DL model. More precisely, I am using Iterator.get_next() to generate tf.tensors  of *batch_size* images and their corresponding labels at each training step and evaluate them to generate the numpy arrays.



##### Q1: Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2? If so, what? If not, is there anything that you can imagine changing in the future?

I didn't change anything from the Part 1 because I use the np.array generated in Part 1 to create a tf.data.Dataset object. In the future, I will need to split the dataset into a training, validation and test set (or some cross-validation). I will also normalize the datasets. I could transform the Dataset into a new Dataset by chaining method calls on the tf.data.Dataset object.
 
 
#####Q2: How do you/did you verify that the pipeline was working correctly?
 
I created a function called **get\_train\_batches()** in the model object to generate a list of numpy arrays of images and labels for **epochs**. I checked that these arrays were both of shape **batchsizeX256X256**.

I also used a label corresponding to the image index in the original dataset to verify that the batches were samples randomly and without replacement in every epochs.
 
#####Q3: Given the pipeline you have built, can you see any deficiencies that you would change if you had more time? If not, can you think of any improvements/enhancements to the pipeline that you could build in?

Instead of first loading the np.arrays and then building the dataset, I could only load the path to the images. Indeed, if the dataset if too big, I won't be able to load the numpy array into memory. In this case I would need to load only the filenames and load the images at each training step.

In Part 1, I was talking about the value inputation to avoid wasting 90% of the data, by inputing the missing labels using neighboring slides in the image.
 