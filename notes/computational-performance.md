# Computational Performance
Notes taken while reading [this](https://d2l.ai/chapter_computational-performance/index.html).

## Compilers and Interpreters
The differences between imperative (interpreted) programming and symbolic programming are as follows:
- Imperative programming is easier. When imperative programming is used in Python, the majority of the code is straightforward and easy to write. It is also easier 
to debug imperative programming code. This is because it is easier to obtain and print all relevant intermediate variable values, or use Python’s built-in 
debugging tools. Although imperative programming is convenient, it may be inefficient.
- Symbolic programming is more efficient and easier to port. It makes it easier to optimize the code during compilation, while also having the ability to port the
program into a format independent of Python. This allows the program to be run in a non-Python environment, thus avoiding any potential performance issues related
to the Python interpreter. It also ensures better optimisation since the entire code is available before execution of the program. 

Historically most deep learning frameworks choose between an imperative or a symbolic approach. For example, Theano, TensorFlow (inspired by the latter), Keras 
and CNTK formulate models symbolically. Conversely, Chainer and PyTorch take an imperative approach. An imperative mode was added to TensorFlow 2.0 (via Eager) 
and Keras in later revisions.

When designing Gluon, developers considered whether it would be possible to combine the benefits of both programming models. This led to a hybrid model that lets 
users develop and debug using pure imperative programming, while having the ability to convert most programs into symbolic programs to be run when product-level 
computing performance and deployment are required. For an example loof at the reference. MXNet is able to combine the advantages of both approaches as needed.
Models constructed by the HybridSequential and HybridBlock classes are able to convert imperative programs into symbolic programs by calling the hybridize method.

## Asynchronous Computation 
Today’s computers are highly parallel systems, consisting of multiple CPU cores (often multiple threads per core), multiple processing elements per GPU and often 
multiple GPUs per device. In short, we can process many different things at the same time, often on different devices. Unfortunately Python is not a great way of 
writing parallel and asynchronous code, at least not with some extra help. After all, Python is single-threaded and this is unlikely to change in the future.
