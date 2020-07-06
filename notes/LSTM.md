# LSTMs

From "Hands on Deep Learning with Python - Sudharsan Ravichandiran".

The drawback of a recurrent neural network (RNN) is that it will not retain information for a long time in memory. We know that an RNN stores sequences of 
information in its hidden state but when the input sequence is too long, it cannot retain all the information in its memory due to the vanishing gradient problem.

To combat this, we introduce a variant of RNN called a long short-term memory (LSTM) cell, which resolves the vanishing gradient problem by using a special 
structure called a gate. Gates keep the information in memory as long as it is required. They learn what information to keep and what information to discard 
from the memory. Basically, RNN cells are replaced with LSTM cells in the hidden units.

A typical LSTM cell consists of three special gates called the input gate, output gate, and forget gate. These three gates are responsible for deciding what 
information to add, output, and forget from the memory. 

In an RNN cell, we used the hidden state, for two purposes: one for storing the information and the other for making predictions. Unlike RNN, in the LSTM 
cell we break the hidden states into two states, called the cell state and the hidden state. 
- The cell state is also called internal memory and is where all the information will be stored
- The hidden state is used for computing the output, that is, for making predictions

Both the cell state and hidden state are shared across every time step.

## Forget Gate

The forget gate, is responsible for deciding what information should be removed from the cell state (memory).

The forget gate is controlled by a sigmoid function. At time step, we pass input, and the previous hidden state to the forget gate. 
- It returns 0 to forget info from cell state and 1 to retain info

```
output_forget_gate = sigmoid(input*w1 + hidden_(i-1)*w2 + bias)
```

## Input gate

The input gate is responsible for deciding what information should be stored in the cell state. 
Similar to the forget gate, the input gate is controlled by a sigmoid function that returns output in the range of 0 to 1(1 to store).

The mathematical formula for the input gate is the same as the forget gate. Only that it learns to do something different. 

## Output Gate

We will have a lot of information in the cell state (memory). The output gate is responsible for deciding what information should be taken from the cell state 
to give as an output. The output gate will look up all the information in the cell state and select the correct information to fill the blank. 

Similar to other gates, it is also controlled by a sigmoid function, and its form is same as well. 

















