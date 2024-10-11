# %%
import sys, os
sys.path.append(os.path.abspath('../workspace'))
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
from functools import partial

# %%

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
X = state[:,:-1]
Y = state[:,-1:]
data = {'train': 
            {'X': X
            ,'Y': Y
            ,'trial_type': None
            ,'trial_switch': None
            ,'loss_mask': None
            },
        'test': 
            {'X': X
            ,'Y': Y
            ,'trial_type': None
            ,'trial_switch': None
            ,'loss_mask': None
            }
        }

# %%
n_in_: int = 2
n_h_: int = 16
n_out_: int = 1
alpha_ = 1
activation_ = f.relu


#%% 

def initializeParametersIO(n_in: int, n_h: int, n_out: int
                        ) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    W_in = np.random.normal(0, np.sqrt(1/(n_in)), (n_h, n_in))
    W_rec = np.linalg.qr(np.random.normal(0, 1, (n_h, n_h)))[0]
    W_out = np.random.normal(0, np.sqrt(1/(n_h)), (n_out, n_h))
    b_rec = np.random.normal(0, np.sqrt(1/(n_h)), (n_h, 1))
    b_out = np.random.normal(0, np.sqrt(1/(n_out)), (n_out, 1))

    W_rec_ = torch.nn.Parameter(torch.tensor(W_rec, requires_grad=True, dtype=torch.float32))
    W_in_ = torch.nn.Parameter(torch.tensor(W_in, requires_grad=True, dtype=torch.float32))
    b_rec_ = torch.nn.Parameter(torch.tensor(b_rec, requires_grad=True, dtype=torch.float32))
    W_out_ = torch.nn.Parameter(torch.tensor(W_out, requires_grad=True, dtype=torch.float32))
    b_out_ = torch.nn.Parameter(torch.tensor(b_out, requires_grad=True, dtype=torch.float32))

    return W_rec_, W_in_, b_rec_, W_out_, b_out_

A = TypeVar('A')  # T can be any type
B = TypeVar('B')  # T can be any type
C = TypeVar('C')  # T can be any type
apply2: Callable[[Callable[[A, B], C], B, C], C] = lambda fn, x, y: fn(x, y)


# %%



# T = TypeVar('T')  # T can be any type
# H = TypeVar('H')
# P = TypeVar('P')
# X = TypeVar('X')


# # def readOut(hiddenState: H, readout: Callable[[H], T]) -> T:  # Just called function app $. 1D case of map
# #     return readout(hiddenState)

# def recurrence(stateTransform: Callable[[H, P], H], parameterTransform: Callable[[H, P], P]) -> Callable[[H, P], tuple[H, P]]:
#     def transform(state: H, parameter: P) -> tuple[H, P]:
#         sprime = stateTransform(state, parameter)
#         return sprime, parameterTransform(sprime, parameter)
#     return transform

# def recurFixedPoint(transform: Callable[[H, P], tuple[H, P]]) -> Callable[[tuple[H, P]], Generator[tuple[H, P], None, None]]:
#     def fix(s: H, p: P) -> Generator[tuple[H, P], None, None]:  # make transform a fixed point function
#         yield (s, p) 
#         s, p = transform(s, p)
#         yield from fix(s, p)
#     return fix

# makeRecurrence: Callable[[Callable[[H], H], Callable[[P], P]], Callable[[tuple[H, P]], Generator[tuple[H, P], None, None]]] \
#     = lambda st, pt: recurFixedPoint(recurrence(st, pt))


# def readoutRecurrence(readout: Callable[[H], T]) -> Callable[[Generator[tuple[H, P], None, None]], Iterator[tuple[H, P, T]]]:
#     def curryGenerator(sequence: Generator[tuple[H, P], None, None]) -> Iterator[tuple[H, P, T]]:
#         def m(tup):
#             s, p = tup 
#             return (s, p, readout(s))
#         return map(m, sequence)
#     return curryGenerator


# s = 1
# p = 2
# st = lambda x, y: x+y
# pt = lambda x, y: 2*y - x 
# readout = lambda x: x
# zs = readoutRecurrence(readout)(makeRecurrence(st, pt)(s, p))

# %%

# we can always curry 

# M: (h x (h + x + 1))

# %%

def transition(W_in, W_rec, b_rec, activation, alpha, x , h):
    return (1 - alpha) * h + alpha * activation(f.linear(x, W_in, bias=None) + f.linear(h, W_rec, b_rec))






rnn = nn.RNN(n_in, n_h, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)




# %%
rnn = RNN(W_in, W_rec, W_out,
            b_rec, b_out,
            activation=tanh,
            alpha=alpha,
            output=identity,
            loss=mean_squared_error)

lr = 0.001
optimizer = Stochastic_Gradient_Descent(lr=lr)
learn_alg = Future_BPTT(rnn, outT+1)  # plus 1 for index 0

monitors = []

feed = np.arange(int(st), int(et))
state = np.array(map(addMemoryTask, feed))

np.random.seed(1)
sim = Simulation(rnn)
sim.run(data, learn_alg=learn_alg,
                optimizer=optimizer,
                monitors=monitors,
                verbose=True)




# %%
