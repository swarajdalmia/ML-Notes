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

## Updating the cell state

We just learned how all three gates work in an LSTM network, but the question is, how can we actually update the cell state by adding relevant new information and deleting information that is not required from the cell state with the help of the gates?

### Adding information

To hold all the new information that can be added to the cell state (memory), we create a new vector called `g_t`. It is called a candidate state or internal state vector. Unlike gates that are regulated by the sigmoid function, the candidate state is regulated by the tanh function, cause we want the output range to be 
between -1 adn 1, not 0 and 1. 

How do we decide whether the information in the candidate state is relevant? We learned that the input gate is responsible for deciding whether to add new information or not, so if we multiply, its output with `g_t` i.e `i_t*g_t`. If the input gate returns 0 if the information is not required and 1 if the information is required.

### Forgetting information

We learned that the forget gate is used for removing information that is not required in the cell state. So, if we multiply the previous cell state, and forget gate, then we retain only relevant information in the cell state i.e. `f_t*c_(t-1)`

```
c_t = f_t*c_(t-1) + i_t*g_t
```
The above says that the updated cell state is given by the addition of the part that forgets info and the part that adds info. 

## Updating the hidden state 

We learned that the hidden state is used for computing the output, but how can we compute the output? We know that the output gate is responsible for deciding what information should be taken from the cell state to give as output. 

```
h_t = o_t*tanh(c_t)
```
Updated hidden state is obtained by the multiplication of the output state and the tanh of the updates cell state. 


## Final Output 

The final ouput is given my multiplying h_t with the hidden layer to output layer weight. Then a softmax layer is used to arrive at the final output. 
```
y_t = softmax(w*h_t)
```

## Forward Propagation in LSTMs

![An LSTM fineprint](https://github.com/swarajdalmia/ML-Experiments/blob/master/notes/images/lstm.jpeg)

![Formulas for updates](https://github.com/swarajdalmia/ML-Experiments/blob/master/notes/images/lstm-equations.jpeg)


## Backpropagation 

We compute the loss at each time step to determine how well our LSTM model is predicting the output. Our final loss is the sum of loss at all time steps.
We minimize the loss using gradient descent. We find the derivative of loss with respect to all of the weights used in the network and find the optimal weights 
to minimize the loss. 



