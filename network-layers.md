# Network layers 

## Convolutional layers vs Fully Connected layers 
Advanatges of convolutional layers:
- parameter sharing : a feature detector thats useful in one part is propabaly useful in another part. This is seen in the fact that in conv layers one, moves the
same kerbel around the entire image instead of using completely different weights for each pixel to pixel. 
- sparcity of connections

[Why use convolutional layers](https://www.youtube.com/watch?v=ay3zYUeuyhU&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF&index=11)

## Pooling layers

It serves the dual purposes of mitigating the sensitivity of convolutional layers to location and of spatially downsampling representations.
[read](https://d2l.ai/chapter_convolutional-neural-networks/pooling.html).

A limitation of the feature map output of convolutional layers is that they record the precise position of features in the input. This means that small 
movements in the position of the feature in the input image will result in a different feature map. This can happen with re-cropping, rotation, shifting, 
and other minor changes to the input image.

A problem with the output feature maps is that they are sensitive to the location of the features in the input. One approach to address this sensitivity is to 
down sample the feature maps. This has the effect of making the resulting down sampled feature maps more robust to changes in the position of the feature in the 
image, referred to by the technical phrase “local translation invariance.” Pooling layers provide an approach to down sampling feature maps by summarizing the 
presence of features in patches of the feature map [here](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/).

- interesting : [Global avergae pooling](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/)

## Batch Norm

Training deep neural nets is difficult. And getting them to converge in a reasonable amount of time can be tricky. In this section, we describe batch 
normalization (BN) (Ioffe & Szegedy, 2015), a popular and effective technique that consistently accelerates the convergence of deep nets. Together with 
residual blocks BN has made it possible for practitioners to routinely train networks with over 100 layers.

Challenges in training deep networks:
- Deeper networks are complex and easily capable of overfitting. This means that regularization becomes more critical.
- For a typical MLP or CNN, as we train, the activations in intermediate layers may take values with widely varying magnitudes—both along the layers from the input
to the output, across nodes in the same layer, and over time due to our updates to the model’s parameters. The inventors of batch normalization postulated 
informally that this drift in the distribution of activations could hamper the convergence of the network. Intuitively, we might conjecture that if one layer has 
activation values that are 100x that of another layer, this might necessitate compensatory adjustments in the learning rates.

### How it works ?

Batch normalization is applied to individual layers (optionally, to all of them) and works as follows: In each training iteration, we first normalize the inputs 
(of batch normalization) by subtracting their mean and dividing by their standard deviation, where both are estimated based on the statistics of the current 
minibatch. Next, we apply a scaling coefficient and a scaling offset. It is precisely due to this normalization based on batch statistics that batch normalization
derives its name. Note: that we add a small constant  e>0  to the variance estimate to ensure that we never attempt division by zero.  You might think that this 
noisiness should be a problem. As it turns out, this is actually beneficial.

This turns out to be a recurring theme in deep learning. For reasons that are not yet well-characterized theoretically, various sources of noise in optimization 
often lead to faster training and less overfitting.

Note that if we tried to apply BN with minibatches of size  1 , we would not be able to learn anything. That is because after subtracting the means, each hidden 
node would take value  0 ! As you might guess, since we are devoting a whole section to BN, with large enough minibatches, the approach proves effective and stable.
One takeaway here is that when applying BN, the choice of minibatch size may be even more significant than without BN.

### Controversy 

[look here](https://d2l.ai/chapter_convolutional-modern/batch-norm.html).

In the original paper proposing batch normalization, the authors, in addition to introducing a powerful and useful tool, offered an explanation for why it works: 
by reducing internal covariate shift.

Following the success of batch normalization, its explanation in terms of internal covariate shift has repeatedly surfaced in debates in the technical literature 
and broader discourse about how to present machine learning research. In a memorable speech given while accepting a Test of Time Award at the 2017 NeurIPS 
conference, Ali Rahimi used internal covariate shift as a focal point in an argument likening the modern practice of deep learning to alchemy. Subsequently, 
the example was revisited in detail in a position paper outlining troubling trends in machine learning [Lipton & Steinhardt, 2018]. In the technical literature
other authors ([Santurkar et al., 2018]) have proposed alternative explanations for the success of BN, some claiming that BN’s success comes despite exhibiting 
behavior that is in some ways opposite to those claimed in the original paper.

- Can you replace Dropout by Batch Normalization? How does the behavior change?







