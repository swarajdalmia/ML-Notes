# RNNs
In a normal feedforward neural network, each input is independent of other input. But with a sequential dataset, we need to know about the past input 
to make a prediction. Let's suppose that we want to predict the next word in a sentence; to do so, we need to remember the previous words.

Under such circumstances (in which we need to remember the previous input), to make predictions, we use recurrent neural networks (RNNs).

RNNs unlike feed-forward networks, predict the output not only on the basis of the current input but also on the basis of the previous hidden state which 
captures past information/context. 

[RNN's are utoregressive sequence models](https://d2l.ai/chapter_recurrent-neural-networks/sequence.html)

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

## Many-many Arcitecture 
In many-to-many architectures, we map a sequence of input of arbitrary length to a sequence of output of arbitrary length. This architecture has been used in various applications. Some of the popular applications of many-to-many architectures include language translation, conversational bots, and audio generation.

# Bi-directional RNNs
In a bidirectional RNN, we have two different layers of hidden units. Both of these layers connect from the input layer to the output layer. In one layer, the hidden states are shared from left to right, and in the other layer, they are shared from right to left.

What is the use of bidirectional RNNs? In certain cases, reading the input sequence from both sides is very useful. So, a bidirectional RNN consists of two RNNs, one reading the sentence forward and the other reading the sentence backward.

Bidirectional RNNs have been used in various applications, such as part-of-speech (POS) tagging, in which it is vital to know the word before and after the target word, language translation, predicting protein structure, dependency parsing, and more. 

The final output of the bidirectional RNN is the weighted sum of the outputs of the two composing RNNs. 

# Deep RNNs
Simple RNNs have only 1 hidden layer but a deep RNN by defn should have more than one hidden layer. But then, how are the hidden states computed when we have more than one hidden layer? 

When we have an RNN with more than one hidden layer, hidden layers at the later layers will be computed by taking the previous hidden state and the previous layer's output as input,

# Application : seq2seq models
The sequence-to-sequence model (seq2seq) is basically the many-to-many architecture of an RNN. It has been used for various applications because it can map an arbitrary-length input sequence to an arbitrary-length output sequence. Some of the applications of the seq2seq model include language translation, music generation, speech generation, and chatbots. 

In most real-world scenarios, input and output sequences vary in length and seq2seq is designed to handle that.

The architecture of the seq2seq model is very simple. It comprises two vital components, namely an encoder and a decoder. Let's consider the same language translation task. First, we feed the input sentence to an encoder.

The encoder learns the representation(embeddings) of the input sentence which is an embedding. It is also called the thought vector or context vector. Once the encoder learns the embedding, it sends the embedding to the decoder. The decoder takes this embedding (thought vector) as input and tries to construct a target sentence. 

### Encoder 
An encoder is basically an RNN with LSTM or GRU cells. It can also be a bidirectional RNN. We feed the input sentence to an encoder and, instead of taking the output, we take the hidden state from the final time step as the embeddings.

### Decoder
A decoder is also an RNN with LSTM or GRU cells. 
We know that we start off an RNN by initializing its initial hidden state with random values, but for the decoder's RNN, we initialize the hidden state with the thought vector, , generated by the encoder, instead of initializing them with random values.

But when does the decoder stop? Because our output sequence has to stop somewhere, we cannot keep on feeding the predicted output word from the previous time step as an input to the next time step. When the decoder predicts the output word as EOS, this implies the end of the sentence.

### Beam Search 
Methods for predicting variable-length sequences include greedy search, exhaustive search, and beam search.
Beam search strikes a balance between computational overhead and search quality using a flexible beam size.
[ref](https://d2l.ai/chapter_recurrent-modern/beam-search.html).

### Attention
Let's say the input sentence has 10 words; then we would have 10 hidden states. We take a sum of all these 10 hidden states and use it for the decoder to generate the target sentence. However, not all of these hidden states might be helpful in generating a target word at time step t. Some hidden states will be more useful than other hidden states. So, we need to know which hidden state is more important than another at time step   to predict the target word. To get this importance, we use the attention mechanism, which tells us which hidden state is more important to generate the target word at the time step t. Thus, attention mechanisms basically give the importance for each of the hidden states of the encoder to generate the target word at time step t. There are weights assigned to the hidden vectors to give attention to some. First a score is given and then softmax is used to normalise the probabilities. These are then multiplies to the context vector or embedding. 


## Perplexity 
How to measure the sequence model quality. One way is to check how surprising the text is. A good language model is able to predict with high accuracy tokens that what we will see next. [ref](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html)

We might measure the quality of the model by computing  𝑝(𝑤) , i.e., the likelihood of the sequence. Unfortunately this is a number that is hard to understand and difficult to compare. After all, shorter sequences are much more likely to occur than the longer ones.

For historical reasons, scientists in natural language processing prefer to use a quantity called perplexity rather than bitrate. It can be best understood as the harmonic mean of the number of real choices that we have when deciding which word to pick next. Note that perplexity naturally generalizes the notion of the cross-entropy loss. 



