# Dropout
Let us think briefly about what we expect from a good predictive model. We want it to peform well on unseen data. Classical generalization theory suggests that to
close the gap between train and test performance, we should aim for a simple model. Simplicity can come in the form of a small number of dimensions. 
Another useful notion of simplicity is smoothness, i.e., that the function should not be sensitive to small changes to its inputs.

## Regularization by Injecting Noise
In 1995, Christopher Bishop formalized this idea when he proved that training with input noise is equivalent to Tikhonov regularization. This work drew a clear 
mathematical connection between the requirement that a function be smooth (and thus simple), and the requirement that it be resilient to perturbations in the input.

Then, in 2014, Srivastava et al. [Srivastava et al., 2014](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) developed a clever idea for how to 
apply Bishop’s idea to the internal layers of the network, too. Namely, they proposed to inject noise into each layer of the network before calculating the 
subsequent layer during training. They realized that when training a deep network with many layers, injecting noise enforces smoothness just on the input-output 
mapping. Their idea, called dropout, involves injecting noise while computing each internal layer during forward propagation.  The method is called dropout because
we literally drop out some neurons during training. Throughout training, on each iteration, standard dropout consists of zeroing out some fraction (typically 50%) 
of the nodes in each layer before calculating the subsequent layer.

The key challenge then is how to inject this noise. One idea is to inject the noise in an unbiased manner so that the expected value of each layer—while fixing the 
others—equals to the value it would have taken absent noise.

In Bishop’s work, he added Gaussian noise to the inputs to a linear model. In standard dropout regularization, one debiases each layer by normalizing by the 
fraction of nodes that were retained (not dropped out). In other words, dropout with dropout probability  𝑝  is applied as follows:
```
h' = 0 - with prob = p
   = h/(1-p) - others
```
By design, the expectation remains unchanged, i.e., E[h']=h. Intermediate activations h are replaced by a random variable h' with matching expectation.
Since any of the neurons might be dropped, the calculation of the output layer cannot be overly dependent on any one element of the dropout layers. 

### Dropout during testing
Typically, we disable dropout at test time. Given a trained model and a new example, we do not drop out any nodes (and thus do not need to normalize). 
However, there are some exceptions: some researchers use dropout at test time as a heuristic for estimating the uncertainty of neural network predictions: 
if the predictions agree across many different dropout masks, then we might say that the network is more confident. For now we will put off uncertainty estimation 
for subsequent chapters and volumes.

From [here](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html)
