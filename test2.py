from memory_profiler import profile


from itertools import tee

import torch
import torchvision

from func import *
from toolz.curried import accumulate, last, compose, take, map, mapcat
from operator import add
import torchvision.transforms as transforms



 # MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                        train=True, 
                                        transform=transforms.ToTensor(),  
                                        download=True)


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                        batch_size=100, 
                                        shuffle=True)


for _ in train_loader:
    pass 

for _ in train_loader:
    print("hi")
    pass 