Vapnik - Chervonenkis Dimension or Analysis

## What is the concept of shattering and that of VC dimensions ?

A hypothesis class H, can shatter N data points if we can find a hypothesis h element of H, such that h seperates the positive examples from the negative examples,
then H shatterns N points. 

A straight line can shatter 3 points, since for one possible arrnagement of 3 points ((0,0)(1,0)(0,1)) it can shatter all the 8 labellings. 

Note : When we say H shatters N data points, it doesnt mean it has to shatter all possible positions of N data points. But even if it shatters all possible labellings
of only one positional arrnagement, then it said to shatter N. 

VC Dimension of H is the maximum number of points N, that H can shatter. 

Therefore VC dimension of straight lines is 3. (it cant shatter all 16 labellings of any 1 positioning of 4 points). 

VC dimension of a circle centred at zero is 1. 

VC dimension is a measure of the capacity (complexity, expressive power, richness, or flexibility) of a set of functions. It is not correlated with the number 
of parameters necessarily. 


## Statistical Leaning theory

VC diumension also helps provide the upper bound on the difference between the test error and the training error. 
[https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_dimension]

## Uses

To find the right model, instead of using cross validation error one can also use VC based measures to find the model of the right complexity. 
