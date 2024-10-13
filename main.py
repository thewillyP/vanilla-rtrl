# %%
import sys, os


from torch_tools.torch_rnn import Torch_RNN
sys.path.append(os.path.abspath('../workspace'))
from func import fst, snd, scan, reduce_, swap, uncurry
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
from toolz.curried import curry, compose, identity, take_nth, accumulate, apply, map, concat, take, drop
from functools import reduce
import torchvision
import torchvision.transforms as transforms
from itertools import tee


"""
Why does it make sense for my parameters to be a function?
Because the function takes
1) hidden state
2) previous function
3) data under closure
The previous function is literuhly design to
h -> x -> computational graph
Then I can call backprop on the output.
This will give me a new parameter
Which I can then construct a new function out of
and return!
The only thing I needed was for the function to return a 
computational graph. 

so 
fn1 = forward step
fn2 = my forward step + backprop
h0 = initial hidden state
p1 = initial function (randomly initialized weights/hyper)

In RNN case,
my parameters don't transition so literally identity
"""

#%%
T = TypeVar('T')  # T can be any type
H = TypeVar('H')
P = TypeVar('P')
X = TypeVar('X')
Y = TypeVar('Y')
Z = TypeVar('Z')


@curry
def createTransition(fn1: Callable[[H, P, X], H], fn2: Callable[[H, P], P]) -> Callable[[tuple[H, P], X], tuple[H, P]]:
    @curry
    def dualTransition(state: tuple[H, P], x: X) -> tuple[H, P]:
        h0, p0 = state 
        h1 = fn1(h0, p0, x)
        p1 = fn2(h1, p0)
        return h1, p1
    return dualTransition

recurrence = compose(scan, createTransition)  # [hinit, h1, h2, h3, ...]

@curry
def rnnTransition(W_in, W_rec, b_rec, activation, alpha, h, x):
    return (1 - alpha) * h + alpha * activation(f.linear(x, W_in, bias=None) + f.linear(h, W_rec, bias=b_rec))

@curry
def randomWeightQRIO(n: int, m: int):
    return torch.nn.Parameter(torch.tensor(np.linalg.qr(np.random.normal(0, 1, (n, m)))[0], requires_grad=True, dtype=torch.float32))

@curry
def randomWeightIO(shape):
    return torch.nn.Parameter(torch.tensor(np.random.normal(0, np.sqrt(1/shape[-1]), shape), requires_grad=True, dtype=torch.float32))


@curry
def initializeParametersIO(n_in: int, n_h: int, n_out: int
                        ) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    W_in = np.random.normal(0, np.sqrt(1/(n_in)), (n_h, n_in))
    W_rec = np.linalg.qr(np.random.normal(0, 1, (n_h, n_h)))[0]
    W_out = np.random.normal(0, np.sqrt(1/(n_h)), (n_out, n_h))
    b_rec = np.random.normal(0, np.sqrt(1/(n_h)), (n_h,))
    b_out = np.random.normal(0, np.sqrt(1/(n_out)), (n_out,))

    _W_rec = torch.nn.Parameter(torch.tensor(W_rec, requires_grad=True, dtype=torch.float32))
    _W_in = torch.nn.Parameter(torch.tensor(W_in, requires_grad=True, dtype=torch.float32))
    _b_rec = torch.nn.Parameter(torch.tensor(b_rec, requires_grad=True, dtype=torch.float32))
    _W_out = torch.nn.Parameter(torch.tensor(W_out, requires_grad=True, dtype=torch.float32))
    _b_out = torch.nn.Parameter(torch.tensor(b_out, requires_grad=True, dtype=torch.float32))

    return _W_rec, _W_in, _b_rec, _W_out, _b_out

linear_ = curry(lambda w, b, h: f.linear(h, w, bias=b))


def supervisedLoss(   lossFn: Callable[[T, Y], Z]
                    , outputs: Iterator[X]
                    , targets: Iterator[Y]) -> Iterator[Z]:
    return map(lossFn, outputs, targets)  # drop the first readout since . what if n=1

# def supervisedLoss(   xs: Iterator[X]
#                     , ys: Iterator[Y]
#                     , rnnM: Callable[[Iterator[X]], Iterator[T]]
#                     , n: int
#                     , lossFn: Callable[[T, Y], Z]) -> Iterator[Z]:
#     return map(lossFn, compose(drop(1), take_nth(n), rnnM)(xs), ys)  # drop the first readout since . what if n=1

# def supervisedLoss(   zs: Iterator[tuple[X, Y]]
#                     , rnnM: Callable[[Iterator[X]], Iterator[T]]
#                     , n: int
#                     , lossFn: Callable[[T, Y], Z]) -> Iterator[Z]:
#     return map(lossFn, take_nth(n, rnnM(xs)), ys)


# buildRnnLayers: Callable[[tuple[Callable[[X, X], X], X]], Callable[[X], X]] = compose(accumulate(compose), map(uncurry(apply)))

# wrong, it needs to be win, wh1 wh2 wh3 wout
# def createRnnLayers(W_ins, W_recs, b_recs, activation: Callable, alpha: float):
#     @curry
#     def rnnTransition_(act, aph, wins, wrecs, brecs):
#         return rnnTransition(wins, wrecs, brecs, act, aph)
#     return map(rnnTransition_(activation, alpha), W_ins, W_recs, b_recs)


# def randomRNNInitsIO(nlayers: int, n_in: int, n_rec: int, n_out: int):
#     return (randomWeightIO((n_rec, n_in)) for _ in range(nlayers))


# given a [(state1, f: state1 -> state2 -> state1)] -> [state1]
# given a [(f: state1 -> x -> state1, state1)] -> [state1]

# just map apply and then compose functions

# compose(reduce_(lambda s, fn: fn(s), s0), map(lambda s, ffn: ffn(s)))

# p0 = rnnTransition(W_in_, W_rec_, b_rec_, f.relu, alpha_)


#%%

# Hyper-parameters 
# input_size = 784 # 28x28
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

input_size = 28
sequence_length = 28
hidden_size = 128
num_layers = 2

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

alpha_ = 1
activation_ = f.relu


hiddenTransition = lambda h, fp, x: fp(h, x)
# hiddenTransition = lambda h, fp, x: x
parameterTransition = lambda _, fp: fp
getRnnSequence = recurrence(hiddenTransition, parameterTransition)

W_rec_, W_in_, b_rec_, W_out_, b_out_ = initializeParametersIO(input_size, hidden_size, num_classes)
p0 = rnnTransition(W_in_, W_rec_, b_rec_, f.tanh, alpha_)
h0 = torch.zeros(batch_size, hidden_size, dtype=torch.float32)
state0 = (h0, p0)

rnnReadout = linear_(W_out_, b_out_)

i = 0

@curry
def readout(state: tuple[np.ndarray, Callable]) -> Callable[[np.ndarray], np.ndarray]:
    global i
    print(i)
    i+=1
    h, _ = state 
    return rnnReadout(h)



loss = lambda output, target: f.cross_entropy(output, target)
xs_, ys_ = tee(train_loader, 2)
xtream, targets = map(fst, xs_), map(snd, ys_)
getImages = map(lambda image: image.reshape(-1, sequence_length, input_size).permute(1, 0, 2)) # [N, 1, 28, 28] -> [N, 28, 28] -> [28, N, 28]
streamImageRows = compose(concat, getImages) # [28, N, 28] -> turn sequence into stream -> [N, 28] (batch, input vector) where each input vector is a row and 28 rows make an image

outputs = compose(map(readout), drop(1), take_nth(sequence_length), getRnnSequence(state0), streamImageRows)  # rnnModel -> [initial, x1, x2, ...]. drop(1) to skip initial and take every 28th input

lossSequence = supervisedLoss(loss, outputs(xtream), targets)

print(list(take(2, lossSequence)))
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
