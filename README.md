# Behavioral Cloning - Writeup

## Goals

The goals/steps of this project are the following:

- Use the simulator to collect data of good driving behavior.
- Build, a convolution neural network in [Keras](https://keras.io/) that predicts steering angles from images.
- Train and validate the model with a training and validation set.
- Test that the model successfully drives around track one without leaving the road.
- Summarize the results with a written report.

## Rubric points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:

- **model.py** : Contains the script to create and train the model and generate model.h5 file.
- **drive.py** : For driving the car in autonomous mode in the simulator (This is provided [Udacity](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/drive.py).
- **model.h5** : Contains a trained convolution neural network.
- **README.md** : My write up about this project.
- **video.py**: To generate a mp4 video file from the caputred autonomously drived car, provided by Udacity.

#### 2. Submission includes functional code Using the Udacity provided simulator and my drive.py file; the car can be driven autonomously around the track by executing

```
Try:
Python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network using Nvidia architecture. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I tried the [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) model, and the car drove the complete first track after just ten training epochs (this model could be found [here](model.py#L110-L132)).

A model summary is as follows:


| Layer                         |     Description                       |
|:---------------------:|:---------------------------------------------:|
| Input                 | 160x320x3 RGB image                                      A
| Cropping              | Crop top 50 pixels and bottom 20 pixels; output shape = 90x320x3 |
| Normalization         | Each new pixel value = old pixel value/255 - 0.5      |
| Convolution 5x5       | 5x5 kernel, 2x2 stride, 24 output channels, output shape = 43x158x24  |
| RELU                  |                                                       |
| Convolution 5x5       | 5x5 kernel, 2x2 stride, 36 output channels, output shape = 20x77x36   |
| RELU                  |                                                       |
| Convolution 5x5       | 5x5 kernel, 2x2 stride, 48 output channels, output shape = 8x37x48    |
| RELU                  |                                                       |
| Convolution 5x5       | 3x3 kernel, 1x1 stride, 64 output channels, output shape = 6x35x64    |
| RELU                  |                                                       |
| Convolution 5x5       | 3x3 kernel, 1x1 stride, 64 output channels, output shape = 4x33x64    |
| RELU                  |                                                       |
| Flatten               | Input 4x33x64, output 8448    |
| Fully connected       | Input 8448, output 100        |
| Dropout               | Set units to zero with probability 0.5 |
| Fully connected       | Input 100, output 50          |
| Fully connected       | Input 50, output 10           |
| Fully connected       | Input 10, output 1 (labels)   |

#### 2. Attempts to reduce overfitting in the model

I decided to modify the model by applying regularization techniques like [Dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks)). In addition to that, I split the data into training and validation sets to diagnose overfitting, but when I used the fully augmented data set (described in "Creation of the Training Set" below), overfitting did not appear to be a significant problem.  Loss on the validation set was comparable to loss on the test set at the end of training.  Apparently, the (shuffled and augmented) training set was large enough to allow the model to generalize to the validation data as well, even without dropout layers.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually ([model.py line 135](model.py#L135)).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The simulator provides three different images: center, left and right cameras. Each image was used to train and validate the model.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to pre-process the data. I then decided to augment the training dataset by additionally using images from the left and right cameras, as well as a left-right flipped version of the center camera's image. A [new](model.py#L115) `Lambda` layer was introduced to normalize the input images to zero means. This step allows the car to move a bit further, but it didn't get to the first turn. Another `Cropping` [layer](model.py#L113) was introduced, and the first turn was almost there, but not quite.

The second step was to use a powerful model: [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). With the help of more powerful architecture with additional modification like dropout, the performance seems pretty good except the car sometimes could not pass through the sharp turning corner with red safety lines. The only remaining step was to tune the correction applied to the angle associated with the right and left camera images, as described in "Model parameter tuning". I found that the trained network reliably steered the car all the way around the track for several different choices of correction angle. First trial with low value of 0.25 didn't work with my model, causing car to turn too slow. Then I simply increased to see the effect and it actually just worked right away, which was cool.

#### 2. Final Model Architecture

The final model architecture is shown above in "1. An appropriate model architecture has been employed"

#### 3. Creation of the Training Set & Training Process

To have more data, I simply drove the car in the simulator for more than 10 minutes, collecting 5222 images from the first track. 

All these data was used for training the model with three epochs. The data was shuffled randomly. After this training, the car was driving down the road all the time on the [first](run1/output_video.mp4) track. 
