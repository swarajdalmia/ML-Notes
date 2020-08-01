# Few-Shot Learning Algorithms
Deep learning algorithms perform exceptionally well when we have a substantially large dataset. But how can we handle the situation when we don't have a large
number of data points to learn from? In such cases, we can use few-shot learning algorithms, which do not require huge datasets to learn from.

## What is few-shot learning?
Learning from a few data points is called few-shot learning or k-shot learning, where k specifies the number of data points in each of the class in the dataset.
Say we have two classes – apple and orange. When we have exactly one apple and one orange image in our training set, it is called one-shot learning;
So, k in k-shot learning implies the number of data points we have per class. Since there are 2 classes, its called 2-way k-shot learning, and in general 
the term used is n-way, k-shot learning. 

There is also **zero-shot learning**, where we don't have any data points per class. Wait. What? How can we learn when there are no data points at all? In this 
case, we will not have data points, but we will have meta information about each of the class and we will learn from the meta information.

## Siamese Networks
Siamese networks are special types of neural networks and are among the simplest and most popularly used one-shot learning algorithms. 
Siamese networks basically consist of two symmetrical neural networks both sharing the same weights and architecture and both joined together at the end using an
energy function. The objective of our siamese network is to learn whether the two inputs are similar or dissimilar.

Let's say we have two images, A and B, and we want to learn whether the two images are similar or dissimilar. As shown in the following diagram, we feed Image A
to Network and Image B to Network. The role of both of these networks is to generate embeddings(feature vectors) for the input image. So, we can use any network 
that will give us embeddings. Then, we will feed these embeddings to the energy function, which tells us how similar the two input images are. Energy functions are
basically any similarity measure, such as Euclidean distance and cosine similarity.

### Sample Application
For instance, let's say we want to build a face recognition model for our organization and say about 500 people are working in our organization. If we want to 
build our face recognition model using a convolutional neural network (CNN) from scratch then we need many images of all these 500 people, to train the network and
attain good accuracy. But, apparently, we will not have many images for all these 500 people and therefore it is not feasible to build a model using a CNN or any 
deep learning algorithm unless we have sufficient data points. So, in these kinds of scenarios, we can resort to a sophisticated one- shot learning algorithm such 
as a siamese network, which can learn from fewer data points.

Other applications of siamese networks include signature verification, similar question retrieval, and object tracking. 

## Prototypical Networks
Prototypical networks are yet another simple, efficient, and popular learning algorithm. Like siamese networks, they try to learn the metric space to perform 
classification. The basic idea of the prototypical network is to create a prototypical representation of each class and classify a query point (new point) based on
the distance between the class prototype and the query point. The class prototype, is the average of the embeddings of the individual data points in the class.

After finding the distance between the class prototype and query point embeddings, we apply softmax to this distance and get the probabilities. The class that has
high probability will be the class of our query point.

## Relation Networks
Relation networks consist of two important functions: an embedding function and the relation function. The embedding function is used for extracting the features 
from the input. Once we arrive at the embeddings of the querry and the classification classes, feature concatenation is performed and a relation function is then 
applied to arrive at the relative store. As compared to the prototype method, a relation function is used instead of the distance and instead of a prototype, a 
random sample from the class is used.

## Matching Networks
Matching networks are yet another simple and efficient one-shot learning algorithm published by Google's DeepMind.  It can even produce labels for the unobserved 
class in the dataset. It is similar to the above. 









