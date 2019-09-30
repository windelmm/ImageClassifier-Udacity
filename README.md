# Image Classifier

The goal is the train an image classifier using PyTorch which can identify different kinds of flowers.

# Motvation

Udacity Data Scientist Nanodegree project for deep learning module titled as 'Image Classifier with Deep Learning' attempts to train an image classifier to recognize different species of flowers. We can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice we had to train this classifier, then export it for use in our application. We had used a dataset of 102 flower categories.

# Build Status

This project is considered complete.

# Method 
Used torchvision to load the data. The dataset is split into three parts, training, validation, and testing. For the training, applied transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. Also need to load in a mapping from category label to category name. Wrote inference for classification after training and testing the model. Then processed a PIL image for use in a PyTorch model.

# Software and Libraries
This project uses the following software and Python libraries:
NumPy, pandas, scikit-learn, Matplotlib, Pytorch

# Result 
Using the following software and Python libraries: Torch, PIL, Matplotlib.pyplot, Numpy, Seaborn, Torchvision. Thus, achieved an accuracy of 80% on test dataset as a result of above approaches. Performed a sanity check since it's always good to check that there aren't obvious bugs despite achieving a good test accuracy. Plotted the probabilities for the top 5 classes as a bar graph, along with the input image.

# Use

## Command line applications train.py and predict.py

Following arguments mandatory or optional for train.py

* 'data_dir'. 'Provide data directory. Mandatory argument', type = str

* '--save_dir'. 'Provide saving directory. Optional argument', type = str

* '--arch'. 'Vgg13 can be used if this argument specified, otherwise Alexnet will be used', type = str

* '--lrn'. 'Learning rate, default value 0.001', type = float

* '--hidden_units'. 'Hidden units in Classifier. Default value is 2048', type = int

* '--epochs'. 'Number of epochs', type = int

* '--GPU'. "Option to use GPU", type = str


Following arguments mandatory or optional for predict.py

* 'image_dir'. 'Provide path to image. Mandatory argument', type = str

* 'load_dir'. 'Provide path to checkpoint. Mandatory argument', type = str

* '--top_k'. 'Top K most likely classes. Optional', type = int

* '--category_names'. 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str

* '--GPU'. "Option to use GPU. Optional", type = str

# Credits

* Udacity - for providing data and lessons on deep learning

# Author 
Windels Manu
