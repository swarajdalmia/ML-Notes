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
> o3 = torch.as_tensor(data)
> o4 = torch.from_numpy(data)
> print(o1)
> print(o2)
> print(o3)
> print(o4)
tensor([1., 2., 3.])
tensor([1, 2, 3], dtype=torch.int32)
tensor([1, 2, 3], dtype=torch.int32)
tensor([1, 2, 3], dtype=torch.int32)
```
The first definition creates a float tensor and the others create int tensors. The default can vary at times depending on if the system is a 32/64 bit one and by other factors. `torch.Tensor` is a contructor while `torch.tensor` is q factory function but the later has better documentation and configurations as of now. 

`torch.Tensor()` constructor uses the default dtype, `torch.get_default_dtype()`,  when building the tensor. The other calls choose a dtype based on the incoming data. With `torch.Tensor()`, we are unable to pass a dtype to the constructor which is possible with the lower case torch. 

Another big difference, This happens because `torch.Tensor()` and `torch.tensor()` copy their input data while `torch.as_tensor()` and `torch.from_numpy()` share their input data in memory with the original input object. Look [here](https://deeplizard.com/learn/video/AglLTlms7HU) for concrete example. 

#### Best Options 

```
torch.tensor()
torch.as_tensor()  # memory sharing 
```

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


### Tensor Reshaping 

Acessing the shape, rank and num. elements:
```
# Getting the shape of a tensor:
> t.size()
torch.Size([3, 4])

> t.shape
torch.Size([3, 4])

# Rank of a tensor 
> len(t.shape)
2

# no. of elements 
> torch.tensor(t.shape).prod()
tensor(12)

> t.numel()
12
```

Reshaping a tensor can be done by `t.reshape([1,12])`. One can also increase rank while reshaping, as `t.reshape(2,2,3)`.

#### Squeezing/Unsqueezing a Tensor

The next way we can change the shape of our tensors is by squeezing and unsqueezing them.

- Squeezing a tensor `t.squeeze()` removes the dimensions or axes that have a length of one. If there are several dimensions with length 1, it removes all of them.
- Unsqueezing a tensor, `t.unsqueeze(dim = <dim/col where to add the new dimension>)` adds a dimension with a length of one.

#### Flattening a Tensor

Flatenning a tensor, reshapes it to dimensionality one. 

```
def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t
```

Value -1 is passed as the second argument tells the reshape() function to figure out what the value should be based on the number of elements contained within the tensor.

The above can also be done by the predefined function `t.flatten(start_dim=0)`.

#### Concatenating

Concatenating tensors along one of the present dimensions
```
# combine t1 and t2 row-wise (axis-0)
> torch.cat((t1, t2), dim=0)
# combine t1 and t2 col-wise (axis-1)
> torch.cat((t1, t2), dim=1)
# combine t1 and t2 hight-wise (axis-2)
> torch.cat((t1, t2), dim=2)
```

Concatenating a tensors and creating a new dimension 
```
# each of tn are of shape [4,4]
> t = torch.stack((t1, t2, t3))
> t.shape
torch.Size([3, 4, 4])
```
we used the stack() method to concatenate our sequence of three tensors along a new axis.

#### Element-wise Operations

Two tensors must have the same shape in order to perform element-wise operations on them.
Examples:
```
> t1 + t2
> t1 - 2 # also t1.sub(2)
> t / 2  # also t.div(2)
```

**Broadcastign Tensors :**
Broadcasting describes how tensors with different shapes are treated during element-wise operations.
```
# This is equivalent to how t1 + 2 works.
> t1 + torch.tensor(
    np.broadcast_to(2, t1.shape)
    ,dtype=torch.float32) 
```

Broadcastign makes element-wise oeprations between tensors of different shapes possible. 

There is a post in the [TensorFlow.js](https://deeplizard.com/learn/video/6_33ulFDuCg) series that covers broadcasting in greater detail. There is a practical example, and the algorithm for determining how a particular tensor is broadcasted is also covered.

Comparison operators:
```
> t.eq(0) # t is of dim [3,3]. checks for equality
tensor([[True, False, True],
        [False, True, False],
        [True, False, True]])
# t.ge(), t.gt, t.lt()  are some other comparison operators
```

### Tensor Reduction Operations

A reduction operation on a tensor is an operation that reduces the number of elements contained within the tensor.
Examples:
```
> t.sum() # t.sum(dim=1) reduce along particular dimensions.
# t.sum() returns a tensor. If we want to actually get the value as a number, we use the item() tensor method. 
# mean across the first axis, multiple values are returned, and we can access the numeric values by transforming the output tensor into a Python list or a NumPy array.
> t.prod()
> t.mean()
> t.std()

```
 Let’s look now a very common reduction operation used in neural network programming called Argmax. Argmax returns the index location of the maximum value inside a tensor.

With NumPy ndarray objects, we have a pretty robust set of operations for indexing and slicing, and PyTorch tensor objects support most of these operations as well.


