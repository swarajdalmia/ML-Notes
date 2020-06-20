# CUDA

CUDA stands for Compute Unified Device Architecture, and is an extension of the C programming language and was created by nVidia. CUDA allows the programmer to take advantage of the massive parallel computing power of an nVidia graphics card in order to do general purpose computation. 

Graphic cards use several dozen ALU's and  nVidia’s ALUs are fully programmable, which enables us to harness an unprecedented amount of computational power into the programs that we write.

## CUDA is only well suited for highly parallel algorithms

In order to run efficiently on a GPU, you need to have many hundreds of threads. Generally, the more threads you have, the better. Many serial algorithms do have parallel equivalents, but many do not. If you can’t break your problem down into at least a thousand threads, then CUDA probably is not the best solution for you.

## CUDA is extremely well suited for number crunching

The GPU is fully capable of doing 32-bit integer and floating point operations. In fact, it GPUs are more suited for floating point computations, which makes CUDA an excellent for number crunching. Some of the higher end graphics cards do have double floating point units, however there is only one 64-bit floating point unit for every 16 32-bit floating point units.

## CUDA is well suited for large datasets

Most modern CPUs have a couple megabytes of L2 cache because most programs have high data coherency. However, when working quickly across a large dataset, say 500 megabytes, the L2 cache may not be as helpful. 

GPUs unlike CPU's use massive parallel interfaces in order to connect with it’s memory. For example, the GTX 280 uses a 512-bit interace to it’s high performance GDDR-3 memory. This type of interface is approximately 10 times faster than a typical CPU to memory interface. It is worth noting that most nVidia graphics cards do not have more than 1 gigabyte of memory.

## Writing a kernel in CUDA

As stated previously, CUDA can be taken full advantage of when writing in C. This is good news, since most programmers are very familiar with C. 

What wasn’t stated is that all of these threads are going to be executing the very same function, known as a kernel. Understanding what the kernel is and how it works is critical to your success when writing an application that uses CUDA. The idea is that even though all of the threads of your program are executing the same function, all of the threads will be working with a different dataset. Each thread will know it’s own ID, and based off it’s ID, it will determine which pieces of data to work on. Don’t worry, flow control like ‘if, for, while, do, etc.’ are all supported.

One important thing to remember is that your entire program DOES NOT need to be written in CUDA. 
When something extremely computationally intense is needed, your program can simply call the CUDA kernel function you wrote. 
 

Taken from [here](http://supercomputingblog.com/cuda/what-is-cuda-an-introduction/)
