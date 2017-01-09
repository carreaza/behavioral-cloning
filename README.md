Deep learning problem using data learned from a driving simulator and a end-to-end learning technique to drive a car through a track

MODEL:

Network: Used 6 convolutional layers for the images scaled of size 45x45 after taking away the top 60 pixels (sky) and bottomn 40 pixels (car's hood) with a kernel size of 3x3, using dropout, and max-pooling, ending with 4 fully connected layers, with an output layer of a single neuron. I started using comma.ai's model, removing one of the conv layers as it was taking a long time to train in my CPU. However, I later started using VGG's model, somewhat simplified to handle my CPU (only 3 sets of 2xConv layers). This amount of layers worked well as training didn't take too lng, and the car could drive through the track with no problem. I also added a generator to be able to handle big amounts of data using fit_generator().

input (45x45x3) -> 
Lambda layer (normalize data to -0.5 to 0.5) -> 
3x (
	2x ( Conv Layer(3x3 filter, [32,64,128] depth) -> elu ) ->
	Max-pooling (2x2) -> 
	Droput(50%) -> 
    )
Flatten -> 
FCL Dense (512) ->
Dropout (20%) -> 
elu -> 
FCL Dense (64) -> 
Dropout (50%) -> 
elu -> 
FCL Dense (16) -> 
Dropout (50%) -> 
elu -> 
Output Layer (size 1).

Hyperparameters:
number of epochs 8 (using generator to yield 1 image at a time) chose after trying a lower number like 4 and saw that the car didn't behave as expected. 8 number of epochs was enough and didn't take a long time to train.

I used mean squared error as this is a regression problem using the adam optimizer with a learning rate of 1e-4. Using the default adam settings didn't work as the loss increased after a couple of epochs during training. Therefore, I had to lower the learning rate to prevent overfitting.

DATA: The training data was captured fully on test track #1. First, the a few laps done keeping the car in the middle of the road. A couple of laps in the reverse direction were also done. Second, many recovery maneouvers were done to get data of the car leaving the sides of the road. Finally, it was noted during testing that the vehicle had a hard time to leave tough situations, so i got even more data when the vehicle had to do sharp turns.

PROBLEMS WITH DATA AND TRAINING: Due to the nature of the simulator the inputs (steering angle) were very segmented and not continuous. Using the computer's keyboard as an input, there were a lot of '0s' inputs even when the vehicle was turning. Usually a steering wheel is used in a real vehicle and the inputs are very continuous. THis problem caused the training to be fairly difficult. Possible solutions are to smooth the data by filtering/interpolating, use deeper networks and more training data (which is limited by my slow CPU).

UPDATE 2016-Dec-19: Using Udacity's data which should resolve the problems states above. However, as I am using a CPU training times take very long. I will try to train with 3 conv nets this time and also try some image/data augmentation.

UPDATE 2017-Jan-09: I am using Udacity's data along with the new simplified VGG's architecture. Resizing the images to 45x45, and removing the top 60 pixels and bottom 40 pixels, training is more manageable.

VALIDATING MODEL: The accuracy couldn't be used to determine the validity of the model. Thus, the results were used in the tracks and visually inspected to see if the car stayed in between lanes. 
