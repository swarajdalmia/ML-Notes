# GPU's

Q) What tasks are faster on CPU's and which are faster on GPU's ?

In the beginning, the main tasks that were accelerated using GPUs were computer graphics. Hence the name graphics processing unit, but in recent years, they are increasingly used in other area and a new type of programming model called GPGPU or **general purpose GPU computing** is comping up.

CPU's typically have less than 10 cores while GPU's have hundreds of cores.

CPU is designed to handle a wide range of tasks quickly (as measured by CPU clock speed), but are limited in the concurrency of tasks that can be running.

A GPU is designed to quickly render or process high-resolution images and video concurrently. However they are also commonly used for non-graphical tasks such as deep learning and scientific computation(eg. Matrix operation and high dimensional vector operation). 

In simple terms, if an image is provided to CPU and GPU then CPU process each pixel one by one(sequential) whereas GPU process hundreds of pixel together(concurrently/parallelism) with it’s hundreds of cores.

Parallelism is not alone contributing to faster processing there is also the term **memory bandwidth**.

## Memory Bandwidth

First of all, you have to understand that CPUs are latency optimized while GPUs are bandwidth optimized.

You can visualize this as a CPU being a Ferrari and a GPU being a big truck. The task of both is to pick up packages from a random location A and to transport those packages to another random location B. The CPU (Ferrari) can fetch some memory (packages) in your RAM quickly while the GPU (a big truck) is slower in doing that (much higher latency). However, the CPU (Ferrari) needs to go back and forth many times to do its job (location A → pick up 2 packages → location B … repeat) while the GPU can fetch much more memory at once (location A → pick up 100 packages → location B … repeat).

The best CPUs have about 50GB/s while the best GPUs have 750GB/s memory bandwidth. But there is still the latency that may hurt performance in the case of the GPU.

However, memory access latency can be hidden under thread parallelism(read [here](https://medium.com/@tarun.medtiya18/cpu-vs-gpu-for-deep-learning-feca267a9c0b)). 
This is the first step where the memory is fetched from the main memory (RAM) to the local memory on the chip (L1 cache and registers).

## Large and Fast Registers 
The advantage of the GPU is here that it can have a small pack of registers for every processing unit (stream processor, or SM), of which it has many. Thus we can have in total a lot of registered memory, which is very small and thus very fast. 

Keep in mind that the slower memory always dominates performance bottlenecks. If 95% of your memory movements take place in registers (80TB/s), and 5% in your main memory (0.75TB/s), then you still spend most of the time on memory access of main memory (about six times as much).

## Note
People use GPUs to do machine learning because they expect them to be fast. But transferring variables between contexts is slow. So we want you to be 100% certain that you want to do something slow before we let you do it. If the framework just did the copy automatically without crashing then you might not realize that you had written some slow code.

Also, transferring data between devices (CPU, GPUs, other machines) is something that is much slower than computation. It also makes parallelization a lot more difficult, since we have to wait for data to be sent (or rather to be received) before we can proceed with more operations. This is why copy operations should be taken with great care. As a rule of thumb, many small operations are much worse than one big operation. Moreover, several operations at a time are much better than many single operations interspersed in the code (unless you know what you are doing) This is the case since such operations can block if one device has to wait for the other before it can do something else. It is a bit like ordering your coffee in a queue rather than pre-ordering it by phone and finding out that it is ready when you are.

Last, when we print tensors or convert tensors to the NumPy format, if the data is not in main memory, the framework will copy it to the main memory first, resulting in additional transmission overhead. Even worse, it is now subject to the dreaded Global Interpreter Lock that makes everything wait for Python to complete.

## GPU, TPU and NPU's

Google explains the reason they decided to create the TPU. In short, the CPU architecture is based on the von Neumann architecture. Thus, “every single calculation” of the CPU is stored in L1 cache (memory). This creates the “von Neumann bottleneck” when memory must be accessed. The ALU (arithmetic logic units) is a circuit that controls memory access and it executes each transaction “one by one”.  The graphic below shows the relationship between the CPU, ALU, and memory.

The GPU solves this problem by throwing thousands of ALU’s and cores at the problem. However, even though GPUs process thousands of tasks in parallel, the von Neumann bottleneck is still present – one transaction at a time per ALU. Google solved the bottleneck problem inherent in GPU’s by creating a new architecture called systolic array. In this setup, ALU’s are connected to each other in a matrix. They call this arrangement the Matrix Processor. In one particular configuration, there are 32,768 ALU’s. 

[Article](https://www.bizety.com/2019/10/28/ai-chips-gpu-tpu-and-npu/).

Another interesting [article](https://timdettmers.com/2017/08/31/deep-learning-research-directions/) on research directions on Deep learning and computational efficiency.
