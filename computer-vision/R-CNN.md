# From R-CNN to Mask R-CNN

[Source](https://medium.com/@umerfarooq_26378/from-r-cnn-to-mask-r-cnn-d6367b196cfd).

The purpose of R-CNNs(Region Based Convolution Neural Network) is to solve the problem of object detection. Given a certain image, we want to be able to draw bounding boxes over all of the objects. The process can be split into two general components, the region proposal step and the classification step.

The authors note that any class agnostic region proposal method should fit. Selective search is used in particular for RCNN. Search performs the function of generating 2000 different regions that have the highest probability of containing an object. After we’ve come up with a set of region proposals, these proposals are then “warped” into an image size that can be fed into a trained CNN (AlexNet in this case) that extracts a feature vector for each region. This vector is then used as the input to a set of linear SVMs that are trained for each class and output a classification. The vector also gets fed into a bounding box regressor to obtain the most accurate coordinates.

Non-maxima suppression is then used to suppress bounding boxes that have a significant overlap with each other.

## Object detection with R-CNN
It consists of three modules. The first generates category-independent region proposals. These proposals define the set of candidate detection avail-able to detector. The second module is a large convolutional neural network that extracts a fixed-length feature vector from each region. The third module is a set of class- specific linear SVMs.

## Fast R-CNN
Object detection in R-CNN was slow. Fast R-CNN was able to solve the problem of speed by basically sharing computation of the conv layers between different proposals and swapping the order of generating region proposals and running the CNN.  In this model, the image is firstfed through a ConvNet.

An input image and multiple regions of interest (RoIs) are input into a fully convolutional network. Each RoI is pooled into a fixed-size feature map and then mapped to a feature vector by fully connected layers (FCs). The network has two output vectors per RoI: softmax probabilities and per-class bounding-box regression offsets. The architecture is trained end-to-end with a multi-task loss.

## Faster R-CNN
Faster R-CNN works to combat the somewhat complex training pipeline that both R-CNN and Fast R-CNN exhibited. The authors insert a region proposal network (RPN) after the last convolutional layer. This network is able to just look at the last convolutional feature map and produce region proposals from that. Faster R-CNN, is composed of two modules. The first module is a deep fully convolutional network that proposes regions, and the second module is the Fast R-CNN detector that uses the proposed regions.The entire system is a single, unified network for object detection. 

## Mask R-CNN — Extending Faster R-CNN for Pixel Level Segmentation
So far, bounding boxes were identified. Can we extend such techniques to go one step further and locate exact pixels of each object instead of just bounding boxes? Mask R-CNN does this by adding a branch to Faster R-CNN that outputs a binary mask that says whether or not a given pixel is part of an object. 
