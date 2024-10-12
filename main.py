# %%
import sys, os

from torch_tools.torch_rnn import Torch_RNN
sys.path.append(os.path.abspath('../workspace'))
from func import scan
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
from toolz import curry, compose, identity

from functools import reduce



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

@curry
def createTransition(fn1: Callable[[H, P, X], H], fn2: Callable[[H, P], P]) -> Callable[[tuple[H, P], X], tuple[H, P]]:
    @curry
    def dualTransition(state: tuple[H, P], x: X) -> tuple[H, P]:
        h0, p0 = state 
        h1 = fn1(h0, p0, x)
        p1 = fn2(h1, p0)
        return h1, p1
    return dualTransition

recurrence = compose(scan, createTransition)

@curry
def rnnTransition(W_in, W_rec, b_rec, activation, alpha, h, x):
    return (1 - alpha) * h + alpha * activation(f.linear(x, W_in, bias=None) + f.linear(h, W_rec, bias=b_rec))

@curry
def initializeParametersIO(n_in: int, n_h: int, n_out: int
                        ) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    W_in = np.random.normal(0, np.sqrt(1/(n_in)), (n_h, n_in))
    W_rec = np.linalg.qr(np.random.normal(0, 1, (n_h, n_h)))[0]
    W_out = np.random.normal(0, np.sqrt(1/(n_h)), (n_out, n_h))
    b_rec = np.random.normal(0, np.sqrt(1/(n_h)), (n_h,))
    b_out = np.random.normal(0, np.sqrt(1/(n_out)), (n_out,))

    _W_rec = torch.nn.Parameter(torch.tensor(W_rec, requires_grad=True, dtype=torch.float64))
    _W_in = torch.nn.Parameter(torch.tensor(W_in, requires_grad=True, dtype=torch.float64))
    _b_rec = torch.nn.Parameter(torch.tensor(b_rec, requires_grad=True, dtype=torch.float64))
    _W_out = torch.nn.Parameter(torch.tensor(W_out, requires_grad=True, dtype=torch.float64))
    _b_out = torch.nn.Parameter(torch.tensor(b_out, requires_grad=True, dtype=torch.float64))

    return _W_rec, _W_in, _b_rec, _W_out, _b_out

linear_ = curry(lambda w, b, h: f.linear(h, w, bias=b))

#%% Setup RNN

n_in_: int = 2
n_h_: int = 16
n_out_: int = 1
alpha_ = 1
activation_ = f.relu


hiddenTransition = lambda h, fp, x: fp(h, x)
parameterTransition = lambda _, fp: fp
getRnnSequence = recurrence(hiddenTransition, parameterTransition)

W_rec_, W_in_, b_rec_, W_out_, b_out_ = initializeParametersIO(n_in_, n_h_, n_out_)
p0 = rnnTransition(W_in_, W_rec_, b_rec_, f.relu, alpha_)
h0 = torch.zeros(1, n_h_, dtype=torch.float64)
state0 = (h0, p0)

rnnReadout = linear_(W_out_, b_out_)
rnnLoss = lambda target: compose(curry(f.cross_entropy), rnnReadout)

@curry
def readout(state):
    h, _ = state 
    rnnReadout(h)

myRnnModel = compose(curry(map)(readout), getRnnSequence(state0))



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
