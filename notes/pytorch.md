# PyTorch

Taken from [here](https://deeplizard.com/learn/video/iTKbyFh-7GM)

PyTorch is a deep learning neural network package for Python. 

## PyTorch and Scientific Computing

It is also a scientific computing package. The scientific computing aspect of PyTorch is primarily a result PyTorch’s tensor library and associated tensor operations.

For example, PyTorch **torch.Tensor** objects that are created from NumPy ndarray objects, share memory. This makes the transition between PyTorch and NumPy very cheap from a performance perspective.

## PyTorch  Nvidia Harware(GPU) and Software(CUDA)

Nvidia designs GPUs, and they have created CUDA as a software platform that pairs with their GPU to help parallelize and speed up computation.

CUDA is essentially a software layer that provides an API for developers. The CUDA toolkit can be downloaded for free and it comes with specialized libraries like cuDNN, the CUDA Deep Neural Network library. With PyTorch, CUDA comes baked in from the start. There are no additional downloads required. All we need is to have a supported Nvidia GPU, and we can leverage CUDA using PyTorch. We don’t need to know how to use the CUDA API directly.

PyTorch is written in Python, C++ and CUDA. So unless one is building PyTorch extensions one doesn't need to know details of the CUDA API. PyTorch sits on top of CUDA and cuDNN.

By default a tensor creation by `t = torch.tensor([1,2,3])`  is done on CPU and operations on this tensor is done on the CPU by default. If one wants to move the tensor to the GPU, one needs to execute `t = t.cuda()`. This helpful since some tasks are faster in a CPU and some are faster in a GPU.

## PyTorch History 

PyTorch was released in October 2016. The connection between PyTorch and this Lua version, called Torch(which existed before and still exists), exists because many of the developers who maintain the Lua version are the individuals who created PyTorch.

Soumith Chintala is credited with bootstrapping the PyTorch project. 

## PyTorch and Research 

PyTorch uses dynamic computational graph. This means that the graph is generated on the fly as the operations are created.
This is in contrast to static graphs that are fully determined before the actual operations occur. Many of the cutting edge research topics in deep learning are requiring or benefiting greatly from dynamic graphs.

## Tensors 

Creating a tensor and some operations : 
```
> dd = [[1,2,3],[4,5,6],[7,8,9]]
> t = torch.tensor(dd)
> type(t)
torch.Tensor
> t.shape
torch.Size([3,3])
> t.reshape(1,9)
tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
```
### Tensor Attributes 

Some attributes that all tensors have 
```
> print(t.dtype)
> print(t.device)
> print(t.layout)
torch.float32
cpu
torch.strided
```
Tensor 1.3 and above allow operations between tensors of differet dtypes. 
One can also declare a device using `tensor.device()` and that can be passed to the tensor contructor to create a tensor on that device. The layout, strided in our case, specifies how the tensor is stored in memory. 

### Creating Tensors : From Data

```
> data = np.array([1,2,3])
> o1 = torch.Tensor(data)
> o2 = torch.tensor(data)
> print(o1)
> print(o2)
tensor([1., 2., 3.])
tensor([1, 2, 3], dtype=torch.int32)
```
The first definition creates a float tensor and the 2nd creates a int tensor. 

### Creating Tensors : Without Data

The dtype of all the tensors below is float.

```
torch.eye(2) # creates a square idenity matrix/tensor of dim (2,2)
torch.zeros([2,2])
torch.ones([2,2])
torch.rand([2,2])  # random mumbers are between 0 and 1
```
### Tensor Input to a NN

A tensor input to a NN has 4 axes [Batch/Num Samples,(color)Channels, Height, Width]. 
Initially the number of channels usually correspond to the color but after undergoing convolution the channels represent feature maps.

PyTorch uses NCHW, and it is the case that TensorFlow and Keras use NHWC by default (it can be configured). Ultimately, the choice of which one to use depends mainly on performance. Some libraries and algorithms are more suited to one or the other of these orderings.







