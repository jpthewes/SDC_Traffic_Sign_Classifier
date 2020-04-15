# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/Class_distribution.png "Class Distribution"
[image2]: ./web_pictures/unscaled/30kph.png "30kph limit"
[image3]: ./web_pictures/unscaled/50limit.png "50kph limit"
[image4]: ./web_pictures/unscaled/road_work.png "Road work"
[image5]: ./web_pictures/unscaled/end_of_limits.jpg "End of limits"
[image6]: ./web_pictures/unscaled/yield.png "Yield"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jpthewes/SDC_Traffic_Sign_Classifier). Also there or in this workspace you can find the HTML output as well as the Ipython notebook.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used basic python functions like len() and the numpy library (in this case np.array.shape) to calculate a summary of statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I explored the dataset by visualizing random pictures in a simple way one after each other. In addition to the picture I printed the label and the data itself as an array. With that I took a look at the data structure and got familiar with it.
In addition, I plotted the spreading of the class distribution over the different datasets:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

To pre-process the data I decided to normalize the images. I did that by dividing each pixel value by 255 and therefore getting a range from 0 to 1 for each pixel in that RGB image. I chose to not convert to grayscale, because traffic signs have siginifant colors which could help in recognizing the specific sign. 
As a first step, I decided to convert the images to grayscale because ...



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model is strongly inspired by the Lenet architecture and consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 , valid padding	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 , valid padding	|
| Flatten				| outputs a 1D tensor of 400 elements			|
| Fully connected		| outputs 1D of 120								|
| RELU					|												|
| Dropout				| in training keep_prob of 0.6					|
| Fully connected		| outputs 1D of 84								|
| RELU					|												|
| Dropout				| in training keep_prob of 0.6					|
| Fully connected		| outputs 1D of 43 (=number of classes)			|
| Softmax				| 												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyperparameters:
EPOCHS = 35 
BATCH_SIZE = 256
train_dropout_keep_rate = 0.6
learning_rate = 0.0007

Furthermore I used the Adam Optimizer to reduce the CrossEntropy of the dataset and therefore train the network to better detect the traffic signs. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95% (calculated while training the model after each epoch)
* test set accuracy of 93% (calculated once after feeling confident with the validation set)

At first I had issues with the validation accuracy only stumbling around 75%. After taking a look and adjusting the pre-processing step this increased to 85-90%. Then I started tweaking the hyperparameters. At first I reduced the learning rate and played with some values. 0.0007 has proven in that process to be promising. I also increased the epoch number to 35. 
A last but important step was to introduce the dropout into the model and tweak with the dropout rate. A value of 0.6 as the probability to keep input for the layers showed to be promising in combination with the other hyperparameters. The dropout layer helped to avoid overfitting on the training set.

The model architecture I chose is close to the Lenet architecture. I believe in this architecture for image processing because the convolutional layers help to recognize patterns in the images and the network can reuse the weights.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

The image of the road work sign might be difficult to classify because other traffic signs interfere with the image. This could confuse the network. 
The 50kph speed limit sign might also be difficult to detect, because this does not have a darker background around the sign as the training and validation images had. Therefore this clean picture is a new observation for the network and I don't expect it to detect this image correctly. Same reasons apply for the end-of-limits sign.
The yield sign might be difficult to recognize because of the dark lighting, but is doable. 
The 30kph speed limit sign should be detected as it has similar backgrounds and lighting.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Before letting the network predict the signs I scaled them down to 32x32x3 and sortet them in an array according to the order of their labels. Within that step I made sure that the datatype is float. Furthermore I normalized the images by dividing the pixel values by 255, just as with the training images. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| End of limits			| End of limits									| 
| 30 km/h	 			| 30 km/h						 				|
| Road Work  			| 30 km/h 										|
| Yield					| Priority road									|
| 50 km/h				| Priority road      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This shows that more training with a higher variety (background, lighting) of images need to be done. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is very sure that this is a end of limits sign (probability of 0.99). The top five soft max probabilities for the other classes can be neglected because they are less than 0.1%.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| End of limits									| 



For the second image the model is not very sure about the classification for the 30kph speed limit but it guessed correctly. It can be seen though that a speed limit in general is very likely.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .12         			| 30kph speed limit								| 
| .097     				| Priority road 								|
| .09					| Speed limit (70km/h)							|
| .08	      			| Wild animals crossing			 				|
| .07				    | Speed limit (80km/h) 							|

For the other 3 images the predictions were incorrect as mentioned above and also the probabilities are very unsure about their prediction. The top 5 softmax prob. for the remaining images are always around 5-15% for each prediction. This is at least a little comforting as the model is not sure about a wrong prediction. Still, further training and model adjustments might need to be done to ensure correct detection. 
The top 5 propbailities and their corresponding labels can be found at the bottom of the Ipython notebook underneath the heading "Output Top 5 Softmax Probabilities For Each Image Found on the Web".

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


