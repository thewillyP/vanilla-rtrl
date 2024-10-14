# %%
import sys, os
import time
from memory_profiler import profile


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
S = TypeVar('S')



@curry
def createTransition(fn1: Callable[[H, P, X], H], fn2: Callable[[H, P], P]) -> Callable[[tuple[H, P], X], tuple[H, P]]:
    @curry
    def dualTransition(state: tuple[H, P], x: X) -> tuple[H, P]:
        h0, p0 = state 
        h1 = fn1(h0, p0, x)
        p1 = fn2(h1, p0)
        return h1, p1
    return dualTransition

# don't want to crack out State Monad in python so resort to code duplication. Also don't want to have to make all my transition functions have to thread state which is unclean
@curry
def createTransitionStateM(fn1: Callable[[H, P, X], H], fn2: Callable[[H, P], P], put: Callable[[S, H, P], tuple[S, H, P]]) -> Callable[[tuple[S, H, P], X], tuple[S, H, P]]:
    @curry
    def dualTransitionStateM(state: tuple[S, H, P], x: X) -> tuple[S, H, P]:
        s_, h0_, p0_ = state 
        s, h0, p0 = put(s_, h0_, p0_)
        h1 = fn1(h0, p0, x)
        p1 = fn2(h1, p0)
        return s, h1, p1
    return dualTransitionStateM


recurrence = compose(scan, createTransition)  # [hinit, h1, h2, h3, ...]
recurrenceStateM = compose(scan, createTransitionStateM)

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
    # W_in = np.random.normal(0, np.sqrt(1/(n_in)), (n_h, n_in))
    # W_rec = np.linalg.qr(np.random.normal(0, 1, (n_h, n_h)))[0]
    # W_out = np.random.normal(0, np.sqrt(1/(n_h)), (n_out, n_h))
    # b_rec = np.random.normal(0, np.sqrt(1/(n_h)), (n_h,))
    # b_out = np.random.normal(0, np.sqrt(1/(n_out)), (n_out,))
    W_in = np.random.uniform(-np.sqrt(1/(n_in)), np.sqrt(1/(n_in)), (n_h, n_in))
    W_rec = np.random.uniform(-np.sqrt(1/(n_h)), np.sqrt(1/(n_h)), (n_h, n_h))
    W_out = np.random.uniform(-np.sqrt(1/(n_h)), np.sqrt(1/(n_h)), (n_out, n_h))
    b_rec = np.random.uniform(-np.sqrt(1/(n_h)), np.sqrt(1/(n_h)), (n_h,))
    b_out = np.random.uniform(np.sqrt(1/(n_h)), np.sqrt(1/(n_h)), (n_out,))

    _W_rec = torch.nn.Parameter(torch.tensor(W_rec, requires_grad=True, dtype=torch.float32))
    _W_in = torch.nn.Parameter(torch.tensor(W_in, requires_grad=True, dtype=torch.float32))
    _b_rec = torch.nn.Parameter(torch.tensor(b_rec, requires_grad=True, dtype=torch.float32))
    _W_out = torch.nn.Parameter(torch.tensor(W_out, requires_grad=True, dtype=torch.float32))
    _b_out = torch.nn.Parameter(torch.tensor(b_out, requires_grad=True, dtype=torch.float32))

    return _W_rec, _W_in, _b_rec, _W_out, _b_out

linear_ = curry(lambda w, b, h: f.linear(h, w, bias=b))


@curry
def supervisions( lossFn: Callable[[T, Y], Z]
                , outputMap: Callable[[Iterator[X]], Iterator[T]]
                , inputs: Iterator[X]
                , targets: Iterator[Y]) -> Iterator[Z]:
    return map(lossFn, outputMap(inputs), targets) 



@curry
def hideStateM(triplet):
    _, h, p = triplet
    return h, p

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
learning_rate = 0.001


hiddenTransition = lambda h, fp, x: fp(h, x)
# hiddenTransition = lambda h, fp, x: x
parameterTransition = lambda _, fp: fp
getRnnSequence = recurrence(hiddenTransition, parameterTransition)



W_rec_, W_in_, b_rec_, W_out_, b_out_ = initializeParametersIO(input_size, hidden_size, num_classes)
p0 = rnnTransition(W_in_, W_rec_, b_rec_, activation_, alpha_)
h0 = torch.zeros(batch_size, hidden_size, dtype=torch.float32)
state0 = (h0, p0)

rnnReadout = linear_(W_out_, b_out_)


@curry
def readout(state: tuple[np.ndarray, Callable]) -> Callable[[np.ndarray], np.ndarray]:
    h, _ = state 
    return rnnReadout(h)



"""
The transition from nth to n+1th hidden state will have the nth hidden state reset to 0 and 
readout_n = f(theta_n, ...)
theta_n = 0 if t == n   <--- Needs to be after readout but before update, so must be in n+1's transition function
theta_n+1 = f(theta_n, ...)
if s = sequence length, then I want the s'th read out. Therefore s = t. 
Then the transition from (n-1) to n must have done: s=n-1 |-> s=n. 

Base case:
If t = 1, then it will always be t=1 before every transition so every transition will be fed 0 which is what we want.
If t = 2, then yep it follows.
We want s0 to start at 0 because the index update applies to the initial state that we skip. 
"""

# class StateThreader:

#     __init__ 

@curry
def stopComputationalGraph(n0, s, h, p):
    return (1, h0, p) if n0 == s else (s+1, h, p)

getRNNSequenceStateM = recurrenceStateM(  hiddenTransition
                                        , parameterTransition
                                        , stopComputationalGraph(sequence_length))
stateM0 = (0, h0, p0)

optimizer = torch.optim.Adam([W_rec_, W_in_, b_rec_, W_out_, b_out_], lr=learning_rate)  


def predict(output, target):
    _, predicted = torch.max(output.data, 1)
    n_samples = target.size(0)
    n_correct = (predicted == target).sum().item()
    return (n_samples, n_correct)

def temp(pair):
    print(pair)
    return 100.0 * pair[1] / pair[0]

accuracy = compose(temp #lambda pair: 100.0 * pair[1] / pair[0]
                    , curry(reduce)(lambda res, pair: (res[0] + pair[0], res[1] + pair[1]))
                    , supervisions(predict))


def separateLabelsIO(loader):
    as_, bs_ = tee(loader, 2)
    return map(fst, as_), map(snd, bs_)

def epochsIO(n: int, loader):
    return (separateLabelsIO(loader) for _ in range(n))




# @profile
def test():

    # with torch.no_grad():  # this makes no computation graph build up. 
    getImages = map(lambda image: image.reshape(-1, sequence_length, input_size).permute(1, 0, 2)) # [N, 1, 28, 28] -> [N, 28, 28] -> [28, N, 28]
    streamImageRows = compose(concat, getImages) # [28, N, 28] -> turn sequence into stream -> [N, 28] (batch, input vector) where each input vector is a row and 28 rows make an image
    outputs = compose(map(readout)
                    , map(hideStateM)
                    , drop(1)
                    , take_nth(sequence_length)
                    , getRNNSequenceStateM(stateM0)  #! Rename to hidden state
                    , streamImageRows)  # rnnModel -> [initial, x1, x2, ...]. drop(1) to skip initial and take every 28th input
    # outputs = compose(map(readout), drop(1), take_nth(sequence_length), getRnnSequence(state0), streamImageRows)  # rnnModel -> [initial, x1, x2, ...]. drop(1) to skip initial and take every 28th input


    loss = lambda output, target: f.cross_entropy(output, target)
    # lossSequence = supervisedLoss(loss, outputs(xtream), targets)
    epochs = epochsIO(num_epochs, train_loader)
    doEpochs = compose(  concat
                        , map(uncurry(supervisions(loss, outputs))))
    

    start = time.time()
    for i, l in enumerate(doEpochs(epochs)):
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
                print (f'Epoch [{1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {l.item():.4f}')
    end = time.time()
    print(end - start)

    # Get prediction accuracy
    with torch.no_grad():
        xs_test, ys_test = tee(test_loader, 2)
        xtream_test, targets_test = map(fst, xs_test), map(snd, ys_test)
        print(accuracy(outputs, xtream_test, targets_test))

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
