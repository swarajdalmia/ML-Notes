# Tensorflow Concepts 

TensorFlow is an open source machine learning framework that Google created and it is used to design, build, and train deep learning models. TensorFlow is a python library that used to implement deep networks and performs numerical computations with the help of tensors and computational graphs/data flow graphs.  In TF2.0, the modern tf.keras API brings the simplicity and ease of use of Keras to the TensorFlow project. One can operate at different levels of abstraction.  

## Tensors 

What is a tensor ? and how does it differ from matrices or numpy arrays ?
All the data that flows through a computational graph is called a tensor.

Tensors in tensorflow/deep learning:
- tensors are a popular way of representing data in deep learning frameworks
- Tensors in tensorflow are implemented as multidimensional data arrays.

Rank of a tensor is the dimensionality of the multi-dim array. 
Shape of a tensor depicts the exact size of the milti dim array.

## Data types

There are 3 data types in tf each of which are tensors.
Now, these are not to be confused with data types of int/float/bool etc. Each of the 3 below, can have any of the int/float/bool etc values that tf supports which encompasses almost everything.

- constants 
```
hello_constant = tf.constant("Hello World!")  # dtype = string
# on printing hello_constant it doesn't display hello world unless it is called in a sess.run(). 
node1 = tf.constant(3.0) # by default, dtype = float32
node2 = tf.constant(3) # by default, dtype = int32
``` 

- place holders : a place holder is a promise to provide a value later.  Values for the placeholders will be fed at runtime. We feed the data to the computational graphs using placeholders.

We assign the values of the placeholder using the feed_dict parameter. The feed_dict parameter is basically the dictionary where the key represents the name of the placeholder, and the value represents the value of the placeholder.

```
# the size of the placeholder is not specified just that it will contain a float tensor. The tensor can be of any rank
a = tf.placeholder(tf.float32)
b = 5*a
with tf.Session() as sess:
    print(sess.run(b,feed_dict={a:[1,7,6,]}))
# feed_dict must be used whenever one wants to assign values to placeholders in python
```

- variables : allows us to add trainable parameters to a graph. Typically used for weights and biases. Names can be assigned to variables. TensorFlow graph, it will be saved as weight. 

```
var = tf.Variable(tf.truncated_normal([3]), name="weight")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #One has to call, glocal_variables_initializer when one uses tf.Varaible
    print(sess.run(var)) # [ 0.50536436 0.56707895 0.98391068 ]
```
We can also initialize a new variable with a value from another variable using initialized_value().
However, after defining a variable, we need to initialize all of the variables in the computational graph. That can be done using tf.global_variables_initializer().

We can also create a TensorFlow variable using tf.get_variable(). It takes the three important parameters, which are name, shape, and initializer.
Unlike tf.Variable(), we cannot pass the value directly to tf.get_variable(); instead, we use initializer. 

Variables created using tf.Variable() cannot be shared, and every time we call tf.Variable(), it will create a new variable. But tf.get_variable() checks the computational graph for an existing variable with the specified parameter. If the variable already exists, then it will be reused; otherwise, a new variable will be created

It is recommended to use tf.get_variable() rather than tf.Variable(), because tf.get_variable, allows you to share variables, and it will make the code refactoring easier.


## Operators 

- Tensorflow has basic math operations in the form of tf.add(), tf.multiply()
- tf.cast() : to convert one datatype into another 

## Computational Graph 

In these graphs, nodes represent mathematical operations, while the edges represent the data, which usually are multidimensional data arrays or tensors. Tensor's flow through the computational graph. 

Tensorflow programs consist of two seperate sections: 
- building a computational graph 
- running the computational graph

The core advantage of having a computational graph is allowing parallelism or dependency driving scheduling which makes training faster and more efficient.

Tensorboard can be used to visualise the computational graph. 

### Executing the computational graph

After making a computational graph one needs to run it. One doesn't get an output for all the variables in the computational graph, unless one specifies them explitly in the running session. To run the entire graph, one calls the last node and tensorflow automatically executes the entire graph including all the assignments and other transformations to the tensors.

### Session 
A session is basically the backbone of a tensorflow program. A session must be called and run to execute/run a computational graph.
```
with tf.Session as sess:
    output,_ = sess.run([variable_name1,var2])
    print(var)

# when one uses with, one doesnt need to explicitly called sess.close() otherwise it is important
```

## Building and using a model 

The steps in using a model using tf.keras can be broken down into 5: [from here](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/). 

- Define the model. Models can be defined either with the Sequential API or the Functional API,
- Compile the model. Reuires selecting the loss function and the optimiser.
```
opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
```
- Fit the model. The most time taking part. Batch size and num. epochs are specified.
```
# verbose = 2(reports model performance after each epoch), verbose = 0(means no output).
model.fit(X, y, epochs=100, batch_size=32, verbose=0)
```
- Evaluate the model.
```
# other metrics along with loss can be evaluated as well
loss = model.evaluate(X, y, verbose=0)
```
- Make predictions.
```
y = model.predict(X)
```

## Types of Models 

Keras is another popularly used deep learning library. It was developed by François Chollet at Google. It is well known for its fast prototyping, and it makes model building simple. It is a high-level library, meaning that it does not perform any low-level operations on its own, such as convolution. It uses a backend engine for doing that, such as TensorFlow. The Keras API is available in tf.keras, and TensorFlow 2.0 uses it as the primary API.

### Sequential Models

The simpler API, which consists in first defining a sequential model and then linearly adding layers to it. 
The visible layer of the network is defined by the “input\_shape” argument on the first hidden layer.

```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_shape=(8,)))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(1))
```

### Functional Models

This one is more flexible and involves connected the output of one layer into the input of another. 

```
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

x_in = Input(shape=(8,))
x = Dense(10)(x_in)
x_out = Dense(1)(x)

model = Model(inputs=x_in, outputs=x_out)
```

## Visualization 

### Model Visualization

- model.summary() : Outputs a text description of the model.
- One can create a plot of the model by calling the plot\_model() function.
```
plot_model(model, 'model.png', show_shapes=True)
```
TensorBoard is TensorFlow's visualization tool, which can be used to visualize a computational graph. It can also be used to plot various quantitative metrics and the results of several intermediate calculations. When we are training a really deep neural network, it becomes confusing when we have to debug the network. So, if we can visualize the computational graph in TensorBoard, we can easily understand such complex models, debug them, and optimize them. 

One can create name scopes in tensor board to reduce complexity and helps us to better understand a model by grouping related nodes together. Having a name scope helps us to group similar operations in a graph. It comes in handy when we are building a complex architecture. 

### Plot Learning Curves

The fit function returns a history object that contains a trace of performance metrics recorded at the end of each training epoch. This includes the chosen loss function and each configured metric, such as accuracy, and each loss and metric is calculated for the training and validation datasets.

A learning curve is a plot of the loss on the training dataset and the validation dataset. We can create this plot from the history object using the Matplotlib library. [Example](https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/)


## Saving and Loading the Model

A model can be saved using the model.save() function. It can be loaded later using the load\_model() function.

The model is saved in H5 format, an efficient array storage format. As such, you must ensure that the h5py library is installed on your workstation.

- The weights can be saved and loaded as well. For details look [here](https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/)
    - the checkpoint binary file contains all the values of the weights, biases, gradients and all the other variables saved.
    - in this case the meta file can be imported to import the meta computational graph. 
    - along with these, tf also has a file named checkpoint which simply keeps a record of latest checkpoint files saved.
- saver.save and saver.restore can be used as well. 


## Eager execution


Eager execution in TensorFlow is more Pythonic and allows for rapid prototyping. Unlike the graph mode, where we need to construct a graph every time to perform any operations, eager execution follows the imperative programming paradigm, where any operations can be performed immediately, without having to create a graph.

In order to enable eager execution, just call the tf.enable_eager_execution() function and we can simply run the below without creating a session and calling sess.run()

```
x = tf.constant(11) 
y = tf.constant(11) 
z = x*y
print z
```


## Tensorflow vs PyTorch

PyTorch is inspired by Torch(Lua based library). It has primarily been developed by Facebook‘s artificial intelligence research group. Both networks use tensors as the fundamental processing type.  


- Static vs Dynamic Approach :
    -  In tensorflow, graphs have to be built before hand(static approach) and are then sent to the GPU where it behaves like a black box. All communication with the outer world is performed via tf.Session object and tf.Placeholder, which are tensors that will be substituted by external data at runtime. TensorFlow provides a way of implementing dynamic graph using a library called TensorFlow Fold, but PyTorch has it inbuilt.  

    - Pytorch has a dynamic approach to graph computation where the graphs are created on the fly. In PyTorch, you can fully dive into every level of the computation, and see exactly what is going on.

- Data Parallelism and Distributed Training
PyTorch has one of the most important features known as declarative data parallelism. This feature allows you to use torch.nn.DataParallel to wrap any module. This will be parallelised over batch dimension and the feature will help you to leverage multiple GPUs easily. PyTorch optimizes performance by taking advantage of native support for asynchronous execution from Python. In TensorFlow, you'll have to manually code and fine tune every operation to be run on a specific device to allow distributed training. 

- Visualization : When it comes to visualization of the training process, TensorFlow takes the lead. Visualization helps the developer track the training process and debug in a more convenient way. TenforFlow’s visualization library is called TensorBoard. PyTorch developers use Visdom, however, the features provided by Visdom are very minimalistic and limited, so TensorBoard scores a point in visualizing the training process.

- Project Deployment is much better/easier with Tensorflow. [Look here](https://builtin.com/data-science/pytorch-vs-tensorflow) or [here](https://towardsdatascience.com/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b).

- Speed : There is not a lot of difference in speed/efficiency.

- PyTorch is easier to debug. Since computation graph in PyTorch is defined at runtime you can use our favorite Python debugging tools such as pdb, ipdb, PyCharm debugger or old trusty print statements. This is not the case with TensorFlow. You have an option to use a special tool called tfdbg which allows to evaluate tensorflow expressions at runtime and browse all tensors and operations in session scope. Of course, you won’t be able to debug any python code with it, so it will be necessary to use pdb separately. [here](https://towardsdatascience.com/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b).

PyTorch is a good choice for research oriented developers. However, for production purposes, TF, seems better.
 

## Learning Tensorlfow/Other Resources

[https://github.com/jtoy/awesome-tensorflow]
