# The Project (Bird's-Eye View)
There are four general steps that we’ll be following as we move through this project:

- Prepare the data
- Build the model
- Train the model
- Analyze the model’s results

## Preparing the data
- Extract – The raw data was extracted from the web.
- Transform – The raw image data was transformed into a tensor.
- Load – The train\_set wrapped by (loaded into) the data loader giving us access to the underlying data.

### Imports

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

# other imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix

import pdb
```
- torch: The top-level PyTorch package and tensor library.
- torch.nn: A subpackage that contains modules and extensible classes for building neural networks.
- torch.optim: A subpackage that contains standard optimization operations like SGD and Adam.
- torch.nn.functional: A functional interface that contains typical operations used for building neural networks like loss functions and convolutions.
- torchvision: A package that provides access to popular datasets, model architectures, and image transformations for computer vision.
- torchvision.transforms: An interface that contains common transforms for image processing.

```
train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()]))
```
To create a DataLoader wrapper for our training set, we do it like this:
```
train_loader = torch.utils.data.DataLoader(train_set
    ,batch_size=1000
    ,shuffle=True)
```
Now, we can leverage the loader for tasks that would otherwise be pretty complicated to implement by hand.

Now, we should have a good understanding of the `torchvision` module that is provided by PyTorch, and how we can use Datasets and DataLoaders in the `PyTorch torch.utils.data` package to streamline ETL tasks.

### PyTorch Datasets And DataLoaders

- Exploring The Data : 
```
> len(train_set)
60000
# Starting with torchvision 0.2.2
> train_set.targets
tensor([9, 0, 0, ..., 3, 0, 5])
```
If we want to see how many of each label exists in the dataset, we can use the PyTorch bincount() function like so:
```
> train_set.targets.bincount()
tensor([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000])
```
Class imbalance is a common problem, but in our case, we have just seen that the Fashion-MNIST dataset is indeed balanced, so we need not worry about that for our project.

- Accesing a batch of images 
```
> display_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10)
```
If shuffle=True, then the batch will be different each time a call to next occurs. 
```
> batch = next(iter(display_loader))
> images, labels = batch

> print('shapes:', images.shape, labels.shape)
shapes: torch.Size([10, 1, 28, 28]) torch.Size([10])
```
The form of tensor is (batch size, number of color channels, image height, image width).

we can use the `torchvision.utils.make\_grid()` to plot multiple images in a grid.

[How To Plot Images Using PyTorch DataLoader](https://deeplizard.com/learn/video/mUueSPmcOBc).

## Building the model(OOP approach) 
To build neural networks in PyTorch, we extend the `torch.nn.Module` PyTorch class `import torch.nn as nn`. Within the nn package, there is a class called Module, and it is the base class for all of neural network modules which includes layers. Even neural networks extend the nn.Module class. This makes sense because neural networks themselves can be thought of as one big layer. 

When we pass a tensor to our network as input, the tensor flows forward though each layer transformation until the tensor reaches the output layer. Every PyTorch nn.Module has a forward() method, and so when we are building layers and networks, we must provide an implementation of the `forward()` method. When we implement the forward() method of our nn.Module subclass, we will typically use functions from the `nn.functional` package

```
Class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # implement the forward pass
        return t
```
Each of our layers extends PyTorch's neural network Module class. For each layer, there are two primary items encapsulated inside, a forward function definition and a weight tensor. PyTorch's neural network Module class keeps track of the weight tensors inside each layer. The code that does this tracking lives inside the nn.Module class, and since we are extending the neural network module class, we inherit this functionality automatically.

- Making and instance of the network and inspecting it. 
```
network = Network()                                    
> print(network)
Network(
    (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
    (conv2): Conv2d(6, 12, kernel_size=(5, 5), stride=(1, 1))
    (fc1): Linear(in_features=192, out_features=120, bias=True)
    (fc2): Linear(in_features=120, out_features=60, bias=True)
    (out): Linear(in_features=60, out_features=10, bias=True)
)
# if this network wasnt an extension of the module class, we would have gotten the output below 
> print(network)
<__main__.Network object at 0x0000017802302FD0>
```

We can override Python’s default string representation using the __repr__ function.
- accessing the networks layers 
```
> network.conv2
Conv2d(6, 12, kernel_size=(5, 5), stride=(1, 1))

> network.conv1.weight

```
To keep track of all the weight tensors inside the network. PyTorch has a special class called Parameter. The Parameter class extends the tensor class, and so the weight tensor inside every layer is an instance of this Parameter class. This is why we see the Parameter containing text at the top of the string representation output.

- The forward function:
```
import torch.nn.functional as F

def forward(self, t):
    # (1) input layer
    t = t

    # (2) hidden conv layer
    t = self.conv1(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)

    # (3) hidden conv layer
    t = self.conv2(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)

    # (4) hidden linear layer
    t = t.reshape(-1, 12 * 4 * 4)
    t = self.fc1(t)
    t = F.relu(t)

    # (5) hidden linear layer
    t = self.fc2(t)
    t = F.relu(t)

    # (6) output layer
    t = self.out(t)
    #t = F.softmax(t, dim=1)
```
However, in our case, we won't use softmax() because the loss function that we'll use, F.cross\_entropy(), implicitly performs the softmax() operation on its input, so we'll just return the result of the last linear transformation.

    
Remember back at the beginning of the series, we said that PyTorch uses a dynamic computational graph. We'll now we're turning it off.

Turning it off isn’t strictly necessary but having the feature turned off does reduce memory consumption since the graph isn't stored in memory. This code will turn the feature off.
```
> torch.set_grad_enabled(False) 
<torch.autograd.grad_mode.set_grad_enabled at 0x17c4867dcc0>
```

- a forward pass through the network 
```
> network = Network()

> sample = next(iter(train_set)) 
> image, label = sample 
> image.shape 
# adds an extra channel
> pred = network(image.unsqueeze(0)) # image shape needs to be (batch_size × in_channels × H × W)

# find the one with the highest probability
> pred.argmax(dim=1)
tensor([7])
```

## Training the Model

- Get batch from the training set.
- Pass batch to network.
- Calculate the loss (difference between the predicted values and the true values).
- Calculate the gradient of the loss function w.r.t the network's weights.
- Update the weights using the gradients to reduce the loss.
- Repeat steps 1-5 until one epoch is completed.
- Repeat steps 1-6 for as many epochs required to reach the minimum loss.

For training we need to set the gradient tracking feature on `> torch.set_grad_enabled(True)`.

calculating the loss
```
> preds = network(images)
> loss = F.cross_entropy(preds, labels) # Calculating the loss
```

Before we calculate the gradients, let's verify that we currently have no gradients inside our conv1 layer. The gradients are tensors that are accessible in the grad (short for gradient) attribute of the weight tensor of each layer.
```
> network.conv1.weight.grad
None
loss.backward() # Calculating the gradients
> network.conv1.weight.grad.shape
torch.Size([6, 1, 5, 5])
```
These gradients are used by the optimizer to update the respective weights. To create our optimizer, we use the torch.optim package that has many optimization algorithm implementations that we can use. We'll use Adam for our example.

- updating the weights 
```
optimizer = optim.Adam(network.parameters(), lr=0.01)
optimizer.step() # Updating the weights
```

- tieing it all together
```
network = Network()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)

for epoch in range(10):

    total_loss = 0
    total_correct = 0

    for batch in train_loader: # Get Batch
        images, labels = batch 

        preds = network(images) # Pass Batch
        loss = F.cross_entropy(preds, labels) # Calculate Loss

        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print(
        "epoch", epoch, 
        "total_correct:", total_correct, 
        "loss:", total_loss
```
Note: annotated the function using the @torch.no\_grad() PyTorch decoration. This is because we want this functions execution to omit gradient tracking.

This is because gradient tracking uses memory, and during inference (getting predictions while not training) there is no need to keep track of the computational graph. The decoration is one way of locally turning off the gradient tracking feature while executing specific functions. Another place where we can use it is with `with torch.no_grad():`.

[Generating a confusion matrix](https://deeplizard.com/learn/video/0LhiS6yu2qQ).

### Visualisation 
[use tensorboard for pytprch](https://deeplizard.com/learn/video/pSexXMdruFM).

### Tuning for hyperparameters 
[build a class to do things efficiently](https://deeplizard.com/learn/video/NSKghk0pcco)
The next chapter refactors code and is also worth looking at.

### Speeding up computation 
- Using num workers to speed up data loading time incase of multiple cores
To speed up the training process, we will make use of the num\_workers optional attribute of the DataLoader class.

The num\_workers attribute tells the data loader instance how many sub-processes to use for data loading. By default, the num\_workers value is set to zero, and a value of zero tells the loader to load the data inside the main process.

This means that the training process will work sequentially inside the main process. After a batch is used during the training process and another one is needed, we read the batch data from disk.

Now, if we have a worker process, we can make use of the fact that our machine has multiple cores. This means that the next batch can already be loaded and ready to go by the time the main process is ready for another batch. This is where the speed up comes from. The batches are loaded using additional worker processes and are queued up in memory.

#### PyTorch and GPU

Pytorch allows us to seamlessly move data to and from our GPU as we preform computations inside our programs. When we go to the GPU, we can use the cuda() method, and when we go to the CPU, we can use the cpu() method.  We can also use the to() method. To go to the GPU, we write to('cuda') and to go to the CPU, we write to('cpu'). The to() method is the preferred way mainly because it is more flexible. We'll see one example using using the first two, and then we'll default to always using the to() variant.

To make use of our GPU during the training process, there are two essential requirements. 
- the data must be moved to the GPU
- the network must be moved to the GPU.

By default, when a PyTorch tensor or a PyTorch neural network module is created, the corresponding data is initialized on the CPU. Specifically, the data exists inside the CPU's memory.  Now, let's create a tensor and a network, and see how we make the move from CPU to GPU.

```
t = torch.ones(1,1,28,28)
network = Network()

# Now, we call the cuda() method and reassign the tensor and network
t = t.cuda()
network = network.cuda()
```
To perform operations between tensors, they must both be with the CPU or th GPU. 
Note: An important consideration of this is that it explains why nn.Module instances like networks don't actually have a device. It's not the network that lives on a device, but the tensors inside the network that live on a device

#### Writing Device Agnostic PyTorch Code
We'll, one of the reasons that the to() method is preferred, is because the to() method is parameterized, and this makes it easier to alter the device we are choosing, i.e. it's flexible!

Implement a device agnostic pyTorch Code. 

## Sequential Models
The Sequential class allows us to build PyTorch neural networks on-the-fly without having to build an explicit class. This make it much easier to rapidly build networks and allows us to skip over the step where we implement theforward() method. When we use the sequential way of building a PyTorch network, we construct the forward() method implicitly by defining our network's architecture sequentially.

There are three ways to create a Sequential model. Let's see them in action.
- The first way to create a sequential model is to pass nn.Module instances directly to the Sequential class constructor.
- The second way to create a sequential model is to create an OrderedDict that contains nn.Module instances. Then, pass the dictionary to the Sequential class constructor.
- The third way of creating a sequential model is to create a sequential instance using an empty constructor. Then, we can use the add\_module() method to add nn.Module instances to the network after it has already been initialize.

Find the examples [here](https://deeplizard.com/learn/video/bH9Nkg7G8S0).



