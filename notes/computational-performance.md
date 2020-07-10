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

Deep learning frameworks such as MXNet and TensorFlow utilize an asynchronous programming model to improve performance (PyTorch uses Python’s own scheduler leading to a different performance trade-off). Hence, understanding how asynchronous programming works helps us to develop more efficient programs, by proactively reducing computational requirements and mutual dependencies. This allows us to reduce memory overhead and increase processor utilization.

MXNet decouples the Python frontend from an execution backend. This allows for fast asynchronous insertion of commands into the backend and associated parallelism.
Asynchrony leads to a rather responsive frontend. However, use caution not to overfill the task queue since it may lead to excessive memory consumption.
It is recommended to synchronize for each minibatch to keep frontend and backend approximately synchronized. Be aware of the fact that conversions from MXNet’s memory management to Python will force the backend to wait until the specific variable is ready. print, asnumpy and item all have this effect. This can be desirable but a carless use of synchronization can ruin performance.

- For Tensorflow and PyTorch look at ways to better optimise code !! What to keep in mind to write asynchronous code and to reduce latency or other operational 
bottlenecks.  HOw to reduce memory usage ? How to optimise data transforms ? 
[https://www.adaltas.com/en/2019/11/15/avoid-deep-learning-bottlenecks/]
[https://www.sicara.ai/blog/tensorflow-tutorial-training-time]
[https://www.sicara.ai/blog/2019-28-10-deep-learning-memory-usage-and-pytorch-optimization-tricks]

Modern systems have a variety of devices, such as multiple GPUs and CPUs. They can be used in parallel, asynchronously. Modern systems also have a variety of resources for communication, such as PCI Express, storage (typically SSD or via network), and network bandwidth. They can be used in parallel for peak efficiency.
The backend can improve performance through through automatic parallel computation and communication.


