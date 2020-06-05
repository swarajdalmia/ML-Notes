# OOP and TensorFlow 

Contains some suggestions and pointers on designing a Tensorflow model with an Object Oriented Programming approach.

## TF Project Template 

Taken from [here](https://github.com/MrGemy95/Tensorflow-Project-Template) which is one of the most well documented/cited repositories on OOP design with TF.

Inspired from this one can structre ones code like this:

- Create a **model class**, which has a built model function and is also responsible for loading and saving the model. 
- The **trainer class**, is responsible for the training(both per batch and the loop for num. epochs).
- A **logger/utils class** which deals with logging the results and misc utils.
- A **config file** which constains things like path details etc. 
- A **data loader** file which takes care of the deta preprocessing/structuring etc. 
- A **main** in which a session is run and is responsible for the pipeline.

Another one to look at, which is inspired from the above but doesn't contain the trainer class is [here](https://github.com/SantoshPattar/ConvNet-OOP). It's an implementation of a convnet for Fashion-MNIST. Has a sample **data loader** which is missing in the earlier one also the way they have defined the model class(which includes training and eval) is clean. 

## Some Interesting ideas from Danijar's Blog

### Building the model

In TF1, the entire graph for the model has been put within the contructor(init), however that approach isn't not very readable/usable, however it can still be done. Another approach that uses a dictionary for the layers and looks cleaner is discussed [here](https://danijar.com/structuring-models/).  

### Methods for various operations 

Just splitting the code is not the best idea, since every time the functions are called, the computational graph would be extended by new code. Therefore it is a good idea to ensure that the operations are added to the graph only when the function is called for the first time. This is called **lazy-loading**. 
To lessen the lines of code in laxy loading design, one can use user defined decorators that make the model look cleaner; some code is also added to ensure proper visualisation of the computational graph([look here](https://danijar.com/structuring-your-tensorflow-models/)). However there might be some issues with saving/loading the model in this case(look at the comments). 

