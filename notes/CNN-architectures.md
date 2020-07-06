# Popular CNN architectures

## LeNet  
The LeNet architecture is one of the classic architectures of a CNN.
It consists of only seven layers(3 convolutional layers, 2 pooling layers, 1 fully connected layer, and 1 output layer).
It uses a 5 x 5 convolution with a stride of 1, and uses average pooling. 

## AlexNet 
AlexNet is a classic and powerful deep learning architecture. It won the ILSVRC 2012 by significantly reducing the error rate from 26% to 15.3%. 
ILSVRC stands for ImageNet Large Scale Visual Recognition Competition, which is one of the biggest competitions focused on computer vision tasks, such as image 
classification, localization, object detection, and more.

ImageNet is a huge dataset containing over 15 million labeled, high-resolution images, with over 22,000 categories. 
AlexNet was designed by Alex Krizhevsky, Geoffrey Hinton, and Ilya Sutskever.
It consists of 5 convolutional layers and 3 fully connected layers.

It uses the ReLU activation function instead of the tanh function, and ReLU is applied after every layer. It uses dropout to handle overfitting, and dropout 
is performed before the first and second fully connected layers. It uses data augmentation techniques, such as image translation, and is trained using 
batch stochastic gradient descent.

## VGGNet
VGGNet is one of the most popularly used CNN architectures. It was invented by the Visual Geometry Group (VGG) at the University of Oxford. It started to get 
very popular when it became the first runner-up of ILSVRC 2014.

It consists of convolutional layers followed by a pooling layer and then again convolutional layers followed by polling layers. It uses 3 x 3 convolution and 
2 x 2 pooling throughout the network. It is referred to as VGG-n, where n corresponds to a number of layers, excluding the pooling and softmax layer. 
An example value of n is 16. The one shortcoming of VGGNet is that it is computationally expensive, and it has over 160 million parameters.

![an example VGGNet architecture](https://github.com/swarajdalmia/ML-Experiments/tree/master/notes/images/VGGNet.jpeg)

## GoogleNet
GoogleNet, also known as inception net, was the winner of ILSVRC 2014. It consists of various versions, and each version is an improved version of the previous 
one. We will explore each version one by one.
