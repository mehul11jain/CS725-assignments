# CS725-assignment
### XOR Dataset
The input X is a list of 2-dimensional vectors. Every example Xi is represented by a 2-dimensional vector [x,y]. The output yi corresponding to the ith
example is either a 0 or 1. The labels follow XOR-like distribution. That is, the first and third quadrant have the same label (yi= 1) and the second
and fourth quadrants have the same label (yi= 0)

There are a total of 10000 points, and the training, validation and test splits contains 7000, 2000 and
1000 points respectively. As discussed in class, the decision boundaries can be learnt exactly for this
dataset using a Neural Network. Hence, if your implementation is correct, the accuracy on the train,
validation and test sets should be close to 100%

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/37892854/70887175-38505180-2003-11ea-9e88-83c82bd52c60.png">
</p>

### MNIST Dataset
We use the MNIST data set which contains a collection of handwritten numerical digits (0-9) as
28x28-sized binary images. Therefore, input X is represented as a vector of size 784 and the number
of output classes is 10 (1 for each digit). In this case, the features are the grayscale image values at
each of the pixels in the image. These images have been size-normalised and centred in a fixed-size
image. MNIST provides a total 70,000 examples, divided into a test set of 10,000 images and a
training set of 60,000 images. In this assignment, we will carve out a validation set of 10,000 images
from the MNIST training set, and use the remaining 50,000 examples for training.


Simple feedforward neural networks (consisting of fully connected layers separated by non-linear
activation functions) can be used to achieve a fairly high accuracy (even over 97%), but achieving this
accuracy might require some careful tuning of the hyperparameters like the number of layers, number
of hidden nodes and the learning rate.

![MNIST Dataset](https://miro.medium.com/max/795/1*VAjYygFUinnygIx9eVCrQQ.png)

