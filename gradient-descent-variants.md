# Gradient Descent and its Variants 

[A more in depth explanation of all the algorithms discussed can be found here](https://d2l.ai/chapter_optimization/index.html).

From : Hands on Deep Learning with Python - Sudharsan Ravichandiran

Gradient descent is a first order optimisation algorithm. First-order optimization means that we calculate only the first-order derivative. 

## Gradient descent vs stochastic gradient descent

to calculate gradient descent we need to find the loss. Say, when there are a a lot of data points, finding the loss function in gradient descent means taking into
account all the point. However that is very inefficeint for most cases. In SGD we just update the parameters of the model after iterating through every single data 
point in our training set, therefore we dont have to wait for the completion of an epoch to update parameters. However, SGD is prone to noise in the data. 

Using mini-batch gradient descent is the balanced method, where updates are made after taking into account a mini-batch. 

- Gradient descent: Updates the parameters of the model after iterating through all the data points in the training set
- Stochastic gradient descent: Updates the parameter of the model after iterating through every single data point in the training set
- Mini-batch gradient descent: Updates the parameters of the model after iterating n number of data points in the training set

## Momentum-based gradient descent

We discuss two new variants of gradient descent, called momentum and Nesterov accelerated gradient.

### Gradient descent with momentum

We basically take a fraction of the parameter update from the previous gradient step and add it to the current gradient step. In physics, momentum keeps an 
object moving after a force is applied. Here, the momentum keeps our gradient moving toward the direction that leads to convergence.

This helps deal with small noises is the graidents which might otherwise interfere with optimal performance of the convergence algorithm. 

- By performing mini-batch gradient descent with momentum helps us to reduce oscillations in gradient steps and attain convergence faster.

To the normal gradient descent update of weights another term is added which represents the momentum with an alpha weight to decide relative importance of momentum.

### Nesterov accelerated gradient

One problem with momentum is that it might miss out the minimum value. 
That is, as we move closer toward convergence (the minimum point), the value for momentum will be high and the momentum actually pushes the gradient 
step high and it might miss out on the actual minimum value by overshooting it. 

The fundamental motivation behind Nesterov momentum is that, instead of calculating the gradient at the current position, we calculate gradients at the 
position where the momentum would take us to, and we call that position the lookahead position. 


## Adaptive Methods of gradient descent 

### Adagrad

In all of the previous methods we learned about, the learning rate was a common value for all the parameters of the network. However Adagrad 
(short for adaptive gradient) adaptively sets the learning rate according to a parameter. 

Usually, parameters irrespective of the value of gradients(high or low) or the frequency of updates have the same learning rate. However what should happen 
is that parameters that arent updated often or by much should have higher learning rates. Adagrad ensures this. 

For the update of each parameter, adagrad, divides the learning rate by the root of the sum of sqaures of all its previous gradient values. 
In a nutshell, in Adagrad, we set the learning rate to a low value when the previous gradient value is high, and to a high value when the past gradient value 
is lower. This means that our learning rate value changes according to the past gradient updates of the parameters.

### Adadelta 

Adadelta is an enhancement of the Adagrad algorithm. In Adagrad, we noticed the problem of the learning rate diminishing to a very low number.
Although Adagrad learns the learning rate adaptively, we still need to set the initial learning rate manually. However, in Adadelta, we don't need
the learning rate at all.

In Adadelta, instead of taking the sum of all the squared past gradients, we can set a window of size and take the sum of squared past gradients 
only from that window, instead of taking values from its entire past. 

However, the problem is that, although we are taking gradients only from within a window, squaring and storing all the gradients from the window 
in each iteration is inefficient. So, instead of doing that, we can take the running average of gradients. And instead of taking a simple running average 
exponentially decaying running average of gradients is taken such that the the past info decays by a certain amount. 

### RMSProp 

Similar to Adadelta, RMSProp was introduced to combat the decaying learning rate problem of Adagrad. Instead of taking the sum of the square of all the past 
gradients, the moving average is calculated in a different way. There is a para eta which is usually set to a value of 0.9.

One of the key issues of adagrad is that the learning rate decreases at a predefined schedule. While this is generally appropriate for convex problems, it might not be ideal for nonconvex ones, such as those encountered in deep learning. Yet, the coordinate-wise adaptivity of Adagrad is highly desirable as a preconditioner. [Tieleman & Hinton, 2012] proposed the RMSProp algorithm as a simple fix to decouple rate scheduling from coordinate-adaptive learning rates.

## Adaptive Moment Estimation(ADAM)

 While reading about RMSProp, we learned that we compute the running average of squared gradients to avoid the diminishing learning rate problem.
 Similar to this, in Adam, we also compute the running average of the squared gradients. However, along with computing the running average of the squared 
 gradients, we also compute the running average of the gradients.
 
 The running average of the gradients and running average of the squared gradients are basically the first and second moments of those gradients. 
 That is, they are the mean and uncentered variance of our gradients, respectively.
 
 - Adamax
 Adamax is a variant of adam which for the calculation of the 2nd momment uses the L^inf norm instead of the L^2 norm. 
 
 - AMSgrad
 It has been noted that, in some settings, Adam fails to attain convergence or reach the suboptimal solution instead of a global optimal solution. 
 This is due to exponentially moving the averages of gradients. In ADAM we are taking an exponential moving average of gradients, however the issue is that we miss out information about the gradients that occur infrequently.
 Tp deal with this issue a change is made and the new optimiser is called AMSgrad. 
 
 - Nadam
 Nadam is another small extension of the Adam method. As the name suggests, here, we incorporate NAG into Adam. 

In particular, [Reddi et al., 2019] show that there are situations where Adam can diverge due to poor variance control. In a follow-up work [Zaheer et al., 2018] proposed a hotfix to Adam, called Yogi which addresses these issues. For gradients with significant variance we may encounter issues with convergence. They can be amended by using larger minibatches or by switching to an improved estimate for s_t . Yogi offers such an alternative.
 
 ## Learning Rate Scheduler
So far we primarily focused on optimization algorithms for how to update the weight vectors rather than on the rate at which they are being updated. Nonetheless, adjusting the learning rate is often just as important as the actual algorithm.

- Most obviously the magnitude of the learning rate matters. If it is too large, optimization diverges, if it is too small, it takes too long to train or we end up with a suboptimal result. We saw previously that the condition number of the problem matters (see e.g., Section 11.6 for details). Intuitively it is the ratio of the amount of change in the least sensitive direction vs. the most sensitive one.
- Secondly, the rate of decay is just as important. If the learning rate remains large we may simply end up bouncing around the minimum and thus not reach optimality. Section 11.5 discussed this in some detail and we analyzed performance guarantees in Section 11.4. In short, we want the rate to decay, but probably more slowly than  (𝑡−12)  which would be a good choice for convex problems.
- Another aspect that is equally important is initialization. This pertains both to how the parameters are set initially (review Section 4.8 for details) and also how they evolve initially. This goes under the moniker of warmup, i.e., how rapidly we start moving towards the solution initially. Large steps in the beginning might not be beneficial, in particular since the initial set of parameters is random. The initial update directions might be quite meaningless, too.

[Look here](https://d2l.ai/chapter_optimization/lr-scheduler.html). Verious policies like factor scheduler, cosine scheduler, warmup etc are discussed.





