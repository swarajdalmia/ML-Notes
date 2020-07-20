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

## Hardware

When we run code on a computer we need to shuffle data to the processors (CPU or GPU), perform computation and then move the results off the processor back to RAM and durable storage. Hence, in order to get good performance we need to make sure that this works seamlessly without any one of the systems becoming a major bottleneck. For instance, if we cannot load images quickly enough the processor will not have any work to do. Likewise, if we cannot move matrices quickly enough to the CPU (or GPU), its processing elements will starve. Finally, if we want to synchronize multiple computers across the network, the latter should not slow down computation. One option is to interleave communication and computation. 

-  Modern SSDs can operate at 100,000 to 500,000 IOPs, i.e., up to 3 orders of magnitude faster than HDDs.
- SSDs store information in blocks (256 KB or larger). They can only be written as a whole, which takes significant time. Consequently bit-wise random writes on SSD have very poor performance. Likewise, writing data in general takes significant time since the block has to be read, erased and then rewritten with new information. By now SSD controllers and firmware have developed algorithms to mitigate this. 
- Deep learning is extremely compute hungry. Hence, to make CPUs suitable for machine learning one needs to perform many operations in one clock cycle. This is achieved via vector units. They have different names: on ARM they are called NEON, on x86 the latest generation is referred to as AVX2 units. A common aspect is that they are able to perform SIMD (single instruction multiple data) operations.
- Of note is a distinction that is often made in practice: accelerators(GUPs, TPUs) are optimized either for training or inference. For the latter we only need to compute the forward pass in a network. No storage of intermediate data is needed for backpropagation. Moreover, we may not need very precise computation (FP16 or INT8 typically suffice). On the other hand, during training all intermediate results need storing to compute gradients. Moreover, accumulating gradients requires higher precision to avoid numerical underflow (or overflow). This means that FP16 (or mixed precision with FP32) is the minimum required. All of this necessitates faster and larger memory (HBM2 vs. GDDR6) and more processing power. For instance, NVIDIA’s Turing T4 GPUs are optimized for inference whereas the V100 GPUs are preferable for training.
- Adding vector units to a processor core allowed us to increase throughput significantly (in the example in the figure we were able to perform 16 operations simultaneously). What if we added operations that optimized not just operations between vectors but also between matrices? This strategy led to Tensor Cores. 
- Vectorization is key for performance. Make sure you are aware of the specific abilities of your accelerator. E.g., some Intel Xeon CPUs are particularly good for INT8 operations, NVIDIA Volta GPUs excel at FP16 matrix-matrix operations and NVIDIA Turing shines at FP16, INT8 and INT4 operations.
- Match your algorithms to the hardware (memory footprint, bandwidth, etc.). Great speedup (orders of magnitude) can be achieved when fitting the parameters into caches.
- Use profilers to debug performance bottlenecks.


