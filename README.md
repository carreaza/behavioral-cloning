# behavioral-cloning
Deep learning problem using data learned from a driving simulator and a end-to-end learning technique to drive a car through a track

MODEL:

Network:
Used 2 convolutional layers for the images of size 160x320 with a kernel size of 3x3, using dropout, and max-pooling in this order, ending with a fully connected layer of 128 nodes, with an output layer of a single neuron. It looks like this:

input (160x320x1) -> Conv Layer(3x3 filter, 32 depth) -> relu -> Conv Layer(3x3 filter, 32 depth) -> relu -> Max-pooling (2x2) -> Dropout (25%) -> FCL (128) -> Dropout (50%) -> Output Layer (size 1).

number of epochs 8
batch size of 100

I used mean squared error as this is a regression problem using the SGD optimizer.

TRAINING:
The training data was captured fully on test track #1. First, the a few laps done keeping the car in the middle of the road. A couple of laps in the reverse direction were also done. Second, many recovery maneouvers were done to get data of the car leaving the sides of the road. Finally, it was noted during testing that the vehicle had a hard time to leave tough situations, so i got even more data when the vehicle had to do sharp turns.


PROBLEMS WITH DATA AND TRAINING:
Due to the nature of the simulator the inputs (steering angle) were very segmented and not continuous. Using the computer's keyboard as an input, there were a lot of '0s' inputs even when the vehicle was turning. Usually a steering wheel is used in a real vehicle and the inputs are very continuous. THis problem caused the training to be fairly difficult. Possible solutions are to smooth the data by filtering/interpolating, use deeper networks and more training data (which is limited by my slow CPU).
