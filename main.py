# %%
import sys, os
import time
from memory_profiler import profile


from torch_tools.torch_rnn import Torch_RNN
sys.path.append(os.path.abspath('../workspace'))
from func import fst, mapTuple1, snd, scan, reduce_, swap, traverseTuple, uncurry
from learning_algorithms import Future_BPTT
from optimizers import Stochastic_Gradient_Descent
import torch 
from torch.nn import functional as f
from torch import nn
import numpy as np
from core import RNN, Simulation
from functions.Function import Function
from functions import *
from gen_data.Add_Memory_Task import *
import matplotlib.pyplot as plt
from typing import TypeVar, Callable, Generic, Generator, Iterator
from toolz.curried import curry, compose, identity, take_nth, accumulate, apply, map, concat, take, drop, mapcat
from functools import reduce
import torchvision
import torchvision.transforms as transforms
from itertools import tee
from scan import *



# Hyper-parameters 
# input_size = 784 # 28x28
num_classes = 10
num_epochs = 1
batch_size = 100

input_size = 28
sequence_length = 28
hidden_size = 128
num_layers = 2

alpha_ = 1
activation_ = f.relu
learning_rate = 0.001


# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                        train=True, 
                                        transform=transforms.ToTensor(),  
                                        download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                        train=False, 
                                        transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=False)


#%% Setup RNN


W_rec_, W_in_, b_rec_, W_out_, b_out_ = initializeParametersIO(input_size, hidden_size, num_classes)
lossFn = lambda h, t: f.cross_entropy(linear_(W_out_, b_out_, h), t)
p0 = rnnTransition(W_in_, W_rec_, b_rec_, activation_, alpha_)
h0 = torch.zeros(batch_size, hidden_size, dtype=torch.float32)
stateM0 = (0, h0, (p0, lossFn))
getHiddenStates = getHiddenStatesStateful(composeST(incrementCounter, resetHiddenStateAt(sequence_length, h0)))

def lossFnIO(step, h, t):
    loss = lossFn(h, t)
    print (f'Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


# I don't want to apply statem0 immed since I might want to start with a trained version isntead




#%%
optimizer = torch.optim.Adam([W_rec_, W_in_, b_rec_, W_out_, b_out_], lr=learning_rate)  


#%%
def predict(output, target):
    _, predicted = torch.max(output.data, 1)
    n_samples = target.size(0)
    n_correct = (predicted == target).sum().item()
    return (n_samples, n_correct)


# accuracy = compose(   lambda pair: 100.0 * pair[1] / pair[0]
#                     , totalStatistic(predict, lambda res, pair: (res[0] + pair[0], res[1] + pair[1])))

#%%


# @profile
def test():

    image2Rows = compose(traverseTuple, mapTuple1(lambda image: image.reshape(-1, sequence_length, input_size).permute(1, 0, 2)))  # [28, N, 28] -> [N, 28] (batch, input vector)
    
    getOutputs = lambda initState: compose(map(hideStateful)
                                        , drop(1)  # non-autnonmous scan returns h0 w/o +input, whose readout we don't care
                                        , take_nth(sequence_length)
                                        , getHiddenStates(initState)  
                                        , mapcat(image2Rows)) 
    
    outputs = getOutputs(stateM0)
    doEpochs = mapcat(outputs)
    epochs = epochsIO(num_epochs, train_loader)

    start = time.time()
    for i, (h, fp) in enumerate(doEpochs(epochs)):  # Ideally I would want to get the new weights, then update my transition function
        if (i+1) % 100 == 0:
            print(h)
            # print (f'Epoch [{1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {l.item():.4f}')
    # end = time.time()
    # print(end - start)

    # # Get prediction accuracy
    # with torch.no_grad():
    #     xs_test, ys_test = tee(test_loader, 2)
    #     xtream_test, targets_test = map(fst, xs_test), map(snd, ys_test)
    #     # print(accuracy(# TODO -> outputs, xs_test, targets_test))

if __name__ == '__main__':
    test()

# print(compose(list, map(lambda x: x.dtype), take(2), streamImageRows)(xtream))
# print(compose(list, take(1), zip(myRnnModel(streamImageRows(xtream)), ystream)))
# print(compose(list, take(1), myRnnModel, streamImageRows)(xtream))
# print(compose(list, take(1), map(type), map(lambda x: x[1][0]), enumerate)(xtream))


# usage of lot of maps implies I can refactor to compose functions and do one big map instead

# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):  
#         # origin shape: [N, 1, 28, 28]
#         # resized: [N, 28, 28]
#         images = images.reshape(-1, sequence_length, input_size).to(device)
#         labels = labels.to(device)
        
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if (i+1) % 100 == 0:
#             print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')




#%%

t1: float = 6
t2: float = 4
a: float = 2
b: float = -1
t1_dur: float = 0.99
t2_dur: float = 0.99
outT: float = 10
st, et = 0., 11.
addMemoryTask: Callable[[float], tuple[float, float, float]] = createAddMemoryTask(t1, t2, a, b, t1_dur, t2_dur, outT)


# %%
feed = np.arange(int(st), int(et))
state = np.array(list(map(addMemoryTask, feed)))
Xs = state[:,:-1]
Ys = state[:,-1:]
data = {'train': 
            {'X': Xs
            ,'Y': Ys
            ,'trial_type': None
            ,'trial_switch': None
            ,'loss_mask': None
            },
        'test': 
            {'X': Xs
            ,'Y': Ys
            ,'trial_type': None
            ,'trial_switch': None
            ,'loss_mask': None
            }
        }









# # %%
# rnn = RNN(W_in, W_rec, W_out,
#             b_rec, b_out,
#             activation=tanh,
#             alpha=alpha,
#             output=identity,
#             loss=mean_squared_error)

# lr = 0.001
# optimizer = Stochastic_Gradient_Descent(lr=lr)
# learn_alg = Future_BPTT(rnn, outT+1)  # plus 1 for index 0

# monitors = []

# feed = np.arange(int(st), int(et))
# state = np.array(map(addMemoryTask, feed))

# np.random.seed(1)
# sim = Simulation(rnn)
# sim.run(data, learn_alg=learn_alg,
#                 optimizer=optimizer,
#                 monitors=monitors,
#                 verbose=True)




# %%
