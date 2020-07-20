# Single Shot MultiBox Detector 
[Source](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab)

Since AlexNet took the research world by storm at the 2012 ImageNet Large-Scale Visual Recognition Challenge (ILSVRC), deep learning has become the go-to method for image recognition tasks, far surpassing more traditional computer vision methods used in the literature. In the field of computer vision, convolution neural networks excel at image classification.

Nowadays, deep learning networks are better at image classification than humans, which shows just how powerful this technique is. However, we as humans do far more than just classify images when observing and interacting with the world. We also **localize** and **classify** each element within our field of view. These are much more complex tasks which machines are still struggling to perform as well as humans.

- R-CNN : In 2014-15(R-CNN(14), Fast R-CNN(2015), Faster R-CNN(2015), Mask R-CNN(2017)), by exploiting some of the leaps made possible in computer vision via CNNs, researchers developed R-CNNs to deal with the tasks of object detection, localization and classification. Broadly speaking, a R-CNN is a special type of CNN that is able to locate and detect objects in images: the output is generally a set of bounding boxes that closely match each of the detected objects, as well as a class output for each detected object. The achievements displayed through this set of work is truly amazing, yet none of these architectures manage to create a real-time object detector. Without going too much into details, the following problems with the above networks were identified:

- Training the data is unwieldy and too long
- Training happens in multiple phases (e.g. training region proposal vs classifier)
- Network is too slow at inference time (i.e. when dealing with non-training data)

Fortunately, in the last few years, new architectures were created to address the bottlenecks of R-CNN and its successors, enabling real-time object detection. The most famous ones are YOLO (You Only Look Once) and SSD MultiBox (Single Shot Detector).

## SSD
The paper about SSD: Single Shot MultiBox Detector (by C. Szegedy et al.) was released at the end of November 2016 and reached new records in terms of performance and precision for object detection tasks, scoring over 74% mAP (mean Average Precision) at 59 frames per second on standard datasets such as PascalVOC and COCO.

- Single Shot: this means that the tasks of object localization and classification are done in a single forward pass of the network
- MultiBox: this is the name of a technique for bounding box regression developed by Szegedy et al. (we will briefly cover it shortly)
- Detector: The network is an object detector that also classifies those detected objects

### Architecture 
As you can see from the diagram above, SSD’s architecture builds on the venerable VGG-16 architecture, but discards the fully connected layers. The reason VGG-16 was used as the base network is because of its strong performance in high quality image classification tasks and its popularity for problems where transfer learning helps in improving results. Instead of the original VGG fully connected layers, a set of auxiliary convolutional layers (from conv6 onwards) were added, thus enabling to extract features at multiple scales and progressively decrease the size of the input to each subsequent layer.

#### Multibox
The bounding box regression technique of SSD is inspired by Szegedy’s work on MultiBox, a method for fast class-agnostic bounding box coordinate proposals. Interestingly, in the work done on MultiBox an Inception-style convolutional network is used. MultiBox’s loss function also combined two critical components that made their way into SSD :
- Confidence Loss: this measures how confident the network is of the objectness of the computed bounding box. Categorical cross-entropy is used to compute this loss.
- Location Loss: this measures how far away the network’s predicted bounding boxes are from the ground truth ones from the training set. L2-Norm is used here.

In MultiBox, the researchers created what we call priors (or anchors in Faster-R-CNN terminology), which are pre-computed, fixed size bounding boxes that closely match the distribution of the original ground truth boxes. In fact those priors are selected in such a way that their Intersection over Union ratio (aka IoU, and sometimes referred to as Jaccard index) is greater than 0.5.


