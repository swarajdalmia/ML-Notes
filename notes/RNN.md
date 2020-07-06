# RNNs

In a normal feedforward neural network, each input is independent of other input. But with a sequential dataset, we need to know about the past input 
to make a prediction. Let's suppose that we want to predict the next word in a sentence; to do so, we need to remember the previous words.

Under such circumstances (in which we need to remember the previous input), to make predictions, we use recurrent neural networks (RNNs).

RNNs unlike feed-forward networks, predict the output not only on the basis of the current input but also on the basis of the previous hidden state which 
captures past information/context. 

## Feed-forward in RNNs
```
hidden_i = activation_func(weight_1*input + weight_2*hidden_(i-1))
```
This hidden_i is the current output of the hidden layer which serves as an RNN block. 

## Backpropagation through time (BPTT)

Backpropagation with RNNs is a little more challenging due to the recursive nature of the weights and their effect on the loss which spans over time.
The loss for the RNN is the sum of the loss for all the time steps(visualise the RNN being rolled out for the set of sequential inputs).

## Vanishing and Exploding Gradients problem 

Sometimes the prediction of output is dependent on data which occurs way earlier in the sequence. However, the calculation of the partial gradient for the terms 
involves a lot of deriavtes and results in vanishing gradients. This makes it difficult for the weights to take into account words that occur at the 
start of a long sequence.

The vanishing gradients problem occurs not only in RNN but also in other deep networks where we use sigmoid or tanh as the activation function. 
So, to overcome this, we can use ReLU as an activation function instead of tanh. However, we have a variant of the RNN called the long short-term memory (LSTM) 
network, which can solve the vanishing gradient problem effectively. 

Similarly, when we initialize the weights of the network to a very large number, the gradients will become very large at every step. While backpropagating, 
we multiply a large number together at every time step, and it leads to infinity. This is called the exploding gradient problem.
We can use gradient clipping to bypass the exploding gradient problem. In this method, we normalize the gradients according to a vector norm (say, L2) 
and clip the gradient value to a certain range.

# Different types of RNN architectures

## One-one architecture 

In a one-to-one architecture, a single input is mapped to a single output(through the hidden layer), and the output from the time step t is fed as an input to the next time step.

For instance, for a text generation task, we take the output generated from a current time step and feed it as the input to the next time step to generate the next word. This architecture is also widely used in stock market predictions.

## One-many Architecture

In a one-to-many architecture, a single input is mapped to multiple hidden states and multiple output values, which means RNN takes a single input and maps it to an output sequence. Although we have a single input value, we share the hidden states across time steps to predict the output.

One such application of this architecture is image caption generation. We pass a single image as an input, and the output is the sequence of words constituting a caption of the image.

## Many-one Architecture 

A many-to-one architecture, as the name suggests, takes a sequence of input and maps it to a single output value(via hidden layers that pass info to the other). One such popular example of a many-to-one architecture is sentiment classification. 

## Many-mnay Arcitecture 

In many-to-many architectures, we map a sequence of input of arbitrary length to a sequence of output of arbitrary length. This architecture has been used in various applications. Some of the popular applications of many-to-many architectures include language translation, conversational bots, and audio generation.
