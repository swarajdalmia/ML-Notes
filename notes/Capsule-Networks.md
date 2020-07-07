# Capsule Networks
Capsule networks (CapsNets) were introduced by Geoffrey Hinton in 2017 to overcome the limitations of convolutional networks.
Hinton stated the following: "The pooling operation used in convolutional neural networks is a big mistake and the fact that it works so well is a disaster."
In addition to that, the team published an algorithm, called dynamic routing between capsules, that allows to train such a network.

[Hinton's talk, what is wrong with CNNs](https://www.youtube.com/watch?v=rTawFwUvnLE&feature=youtu.be)

[A good series expaining capsules](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b)

### What is wrong with pooling operation ?
The pooling operation is used to reduce dimensionality and to remove unwanted information? The pooling
operation makes our CNN representation invariant to small translations in the input. This translation invariance property of a CNN is not always beneficial, 
and can be prone to misclassifications. For example, let's say we need to recognize whether an image has a face; the CNN will look for whether the image has eyes,
a nose, a mouth, and ears. It does not care about which location they are in. If it finds all such features, then it classifies it as a face. This problem will 
become worse when we have a deep network, as in the deep network, the features will become abstract, and it will also shrink in size due to the several pooling 
operations. (but doesn't the CNN also look for structure and how the parts are joined together ?)

An important thing to understand is that higher-level features combine lower-level features as a weighted sum: activation of a preceding layer are multiplied by
the following layer neuron’s weights and added, before being passed to activation non-linearity. Nowhere in this information flow are the relationships between 
features taken into account. This is simply a flaw in the core design of CNNs since they are based on the basic convolution operation applied to scalar values.

To overcome this, Hinton introduced the Capsule network, which consists of capsules instead of neurons. Like a CNN, the Capsule network checks for the presence of
certain features to classify the image, but it also checks the spatial relationship between them. The key to this richer feature representation is the use of 
vectors rather than scalers. 

Let’s review the basic operations of a convolution in a CNN: matrix multiply (i.e scalar waiting), add it all up, scalar-to-scalar mapping (i.e the activation 
function). Here are the steps(from [here](https://towardsdatascience.com/a-simple-and-intuitive-explanation-of-hintons-capsule-networks-b59792ad46b1)):
- scalar weighting of input scalars
- Sum of weighted input scalars
- scalar-to-scalar non-linearity

The change for Capsule networks can be broken down simply by using vectors instead of scalars:
- matrix multiplication of input vectors
- scalar weighting of input vectors
- sum of weighted input vectors
- vector-to-vector non-linearity

![capsule networks](./images/capsule-networks.jpeg)

Vectors help because the help us encode more information, and not just any kind of information, relational and relative information.
Capsules also have the bonus of being able to achieve good accuracy with far less training data. It doesn’t need 50 examples of the same rotated dog; it just needs one with a vector representation which can easily be transformed. By forcing the model to learn the feature variant in a capsule, we may extrapolate possible variants more effectively with less training data.

[Convnets are invariant, not equivariant](https://brandonmorris.dev/2017/11/16/dynamic-routing-between-capsules/)

### Why it took so long ?

The idea is really simple, there is no way no one has come up with it before! And the truth is, Hinton has been thinking about this for decades. The reason why there were no publications is simply because there was no technical way to make it work before. One of the reasons is that computers were just not powerful enough in the pre-GPU-based era before around 2012. Another reason is that there was no algorithm that allowed to implement and successfully learn a capsule network (in the same fashion the idea of artificial neurons was around since 1940-s, but it was not until mid 1980-s when backpropagation algorithm showed up and allowed to successfully train deep networks).

## What is a capsule ?
A capsule is a group of neurons that learn to detect a particular feature in the image; say, eyes. Unlike neurons, which return a scalar, capsules return a vector.
The length of the vector tells us whether a particular feature exists in a given location, and the elements of the vector represent the properties of the features,
such as, position, angle, and so on.

Just like a CNN, capsules in the earlier layers detect basic features including eyes, a nose, and so on, and the capsules in the higher layers detect high-level 
features, such as the overall face. Thus, capsules in the higher layers take input from the capsules in the lower layers. In order for the capsules in the higher 
layers to detect a face, they not only check for the presence of features such as a nose and eyes, but also check their spatial relationships.

##  Dynamic Routing Algorithm Between Capsules
The authors chose an algorithm that encourages “routing by agreement”: capsules in an earlier layer that cause a greater output in the subsequent layer should be encouraged to send a greater portion of their output to that capsule in the subsequent layer.

Before the routing procedure, every capsule in the earlier layer spreads its output evenly to every capsule in the subsequent layer (initial couplings can be learned like weights, but this isn’t done in the paper). During each iteration of the dynamic routing algorithm, strong outputs from capsules in the subsequent layer are used to encourage capsules in the previous layer to send a greater portion of their output

### CapsNet Architecture
First, we take the input image and feed it to a standard convolution layer, and we call the result convolutional inputs. Then, we feed the convolutional inputs to the primary capsules layer and get the primary capsules. Next, we compute digit capsules with primary capsules as input using the dynamic-routing algorithm.
The digit capsules consist of 10 rows, and each of the rows represents the probability of the predicted digit. That is, row 1 represents the probability of the input digit to be 0, row 2 represents the probability of the digit 1, and so on. Since the input image is digit 3 in the preceding image, row 4, which represents the probability of digit 3, will be high in the digit capsules.

It is discussed [here](https://pechyonkin.me/capsules-4/) and [here](https://brandonmorris.dev/2017/11/16/dynamic-routing-between-capsules/)

### Loss Function
The loss function of the Capsule network. The loss function is the weighted sum of two loss functions called margin loss and reconstruction loss.

## Performance on datasets
With only three layers, the CapsNet architecture performed remarkably well. The authors report a 0.25% test error rate on MNIST, which is close to state of the art and not possible with a similarly shallow convnet.

They also performed so experiments on a MultiMNIST data set: two images from MNIST overlapping each other by up to 80%.

CapsNet is also performant on several other data sets. On CIFAR-10, it has a 10.6% error rate (with an ensemble and some minor architecture modifications), which is roughly the same as when convnets were first used on the data set. CapsNet attain 2.7% error on the smallNORB data set, and 4.3% error on a subset of Street View Housing Numbers (SVHN).
