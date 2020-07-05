# Activation Functions 

Activation functions in neural networks help find the relevant features and ignore the noise. The activation functions decide what signal is to be passed on given a certain input signal.

```
output = activation(sum(weights\*inputs)+bias)
```
The activation functions are responsible for forward propagation. Depending on the error, using back-propagation, the values of the weights and the biases are updated. During this update step, the derivative of the activation function is calculated/used. 

- an activation function introduces a certain complexity but that is very important cause without that, the model would simply be a linear model.

The formula for how the updates of the weight is dependent on the activation function and its behavior can be found in pg 28 of the book:  Hands on Deep Learning with Python - Sudharsan Ravichandiran.

## Binary step function 

```
f(x) = 1, x>=a
     = 0, x<a
```

Can be used in cases of a binary classifer. Does not generalise to classes more than two. 

- Gradients are calculated to update the weights and biases during the backprop process. Since the gradient of the function is zero, the weights and biases don’t update. So gradient descent cant be used in this case.

## Linear Function
```
f(x)=ax
```

- The derivative is always a constant, therefore the update factor is the same irrespective of the extent of the error. Also, a NN with linear activation will never be applicable to non-linear tasks.

Not possible to use backpropagation  (gradient descent) to train the model—the derivative of the function is a constant, and has no relation to the input, X. So it’s not possible to go back and understand which weights in the input neurons can provide a better prediction.

A deep architecture siply collapses to a single layer architecture with the linear function due to properties of compositionality of linear functions. 

## Sigmoid

The output vaires between 0-1, therefore it normalises the output of each neuron. 
```
f(x) = 1/(1+e^-x)
```
This function is non linear. The gradient values are significant for range -3 and 3 but the graph gets much flatter in other regions. This implies that for values greater than 3 or less than -3, will have very small gradients. As the gradient value approaches zero, the network is not really learning. The output of the function is of the same sign always(+ve).

It is continuous and differentiable at all places. 

Disadvantages:
- Vanishing gradient—for very high or very low values of X, there is almost no change to the prediction, causing a vanishing gradient problem.
- Outputs not zero centered.
- Computationally expensive

## Tanh 

The tanh function is very similar to the sigmoid function. The only difference is that it is symmetric around the origin. The range of values in this case is from -1 to 1. Thus the inputs to the next layers will not always be of the same sign.
```
tanh(x)=2sigmoid(2x)-1
```
Similar to sigmoid, the tanh function is continuous and differentiable at all points. Compared to the sigmoid function its gradients are steeper.

## Relu(rectified linear unit)

The main advantage of using the ReLU function over other activation functions is that it does not activate all the neurons at the same time.
```
f(x)=max(0,x)
```
ince only a certain number of neurons are activated, the ReLU function is far more computationally efficient when compared to the sigmoid and tanh function. Its gradient is also either 1 or 0. For neurons with output less than zero, the graident is zero. Hence the weights are not updated and they are dead neurons whose output is zero as well, this is called the dying relu problem. Leaky relu takes care of the dead neurons problem. 

## Leaky Relu 

Leaky ReLU function is nothing but an improved version of the ReLU function. For negative values it passes a very small signal. By doing this there are no more dead neurons. 
```
f(x)= 0.01x, x<0
    = x,     x>=0 
```
However, the leaky ReLU does not provide consistent predictions for negative input values.
## Parameterized Relu 

```
f(x) = x, x>=0
    = ax, x<0
```
In leaky relu a is fixed and is equal to 0.01, however here a is a trainable parameter and the best value can be learned. 
The parameterized ReLU function is used when the leaky ReLU function still fails to solve the problem of dead neurons and the relevant information is not successfully passed to the next layer.

## Exponential Linear Unit

Instead of a straight line, ELU uses a log curve for defining negative values. 
```
f(x) = x,   x>=0
    = a(e^x-1), x<0
```

## Swish 

Discovered by researchers at Google. Swish is as computationally efficient as ReLU and shows better performance than ReLU on deeper models.
```
f(x) = x*sigmoid(x) x>=0
f(x) = x/(1-e^-x) x<0
```

The function is smooth and the function is differentiable at all points(unlike relu variants). This is helpful during the model optimization process and is considered to be one of the reasons that swish outoerforms ReLU.

This function is that swich function is not monotonic. This means that the value of the function may decrease even when the input values are increasing. 

## Softmax 

Softmax function is often described as a combination of multiple sigmoids. We know that sigmoid returns values between 0 and 1, which can be treated as probabilities of a data point belonging to a particular class. Hence, it is often used in classification problems(binary or multi class). 

- Useful for output neurons—typically Softmax is used only for the output layer, for neural networks that need to classify inputs into multiple categories.

# Choosing the right activation function : Heuristics 

- Sigmoid functions and their combinations generally work better in the case of classifiers
- Sigmoids and tanh functions are sometimes avoided due to the vanishing gradient problem
- ReLU function is a general activation function and is used in most cases these days
- If we encounter a case of dead neurons in our networks the leaky ReLU function is the best choice
- Always keep in mind that ReLU function should only be used in the hidden layers 




Refs:
- [https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/]
- [https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/]
