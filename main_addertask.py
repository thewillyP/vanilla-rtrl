

# %%
import sys, os
sys.path.append(os.path.abspath('../workspace'))
import time
from memory_profiler import profile
from torch_tools.torch_rnn import Torch_RNN
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
from toolz.curried import curry, compose, identity, take_nth, accumulate, apply, map, concat, take, drop, mapcat, last, cons
from functools import reduce
import torchvision
import torchvision.transforms as transforms
from itertools import tee, cycle
from scan import *
from operator import add



# [1, outT, 2] -> [1, 2], [N, outT, 2] -> [N, 2]


#%%

t1: float = 6
t2: float = 4
a: float = 3
b: float = -1
t1_dur: float = 0.99
t2_dur: float = 0.99
outT: float = 10
st, et = min(outT - t1, outT - t2), outT+1  # +1 to include outT in range()
addMemoryTask: Callable[[float], tuple[float, float, float]] = createAddMemoryTask(t1, t2, a, b, t1_dur, t2_dur, outT)

#%%

sequence_length = et - st
batch_size = 1
num_examples = 50


# %%
# Create dataset
feed = np.arange(int(st), int(et))
state = np.array(list(map(addMemoryTask, feed)))
Xs = torch.Tensor(state[:,:-1]).unsqueeze(1).repeat(1, batch_size, 1)
Ys = torch.Tensor(state[:,-1:]).unsqueeze(1).repeat(1, batch_size, 1)
train_loader = take(sequence_length * num_examples, cycle(zip(Xs, Ys)))
test_loader = take(sequence_length * num_examples, cycle(zip(Xs, Ys)))



#%% Setup RNN

num_epochs = 10

n_out = 1
n_in = 2
n_h = 32

alpha_ = 1
activation_ = f.relu
learning_rate = 0.001


W_rec_, W_in_, b_rec_, W_out_, b_out_ = initializeParametersIO(n_in, n_h, n_out)

readout = linear_(W_out_, b_out_)
optimizer = torch.optim.Adam([W_rec_, W_in_, b_rec_, W_out_, b_out_], lr=learning_rate)  


p0 = rnnTransition(W_in_, W_rec_, b_rec_, activation_, alpha_)
h0 = torch.zeros(batch_size, n_h, dtype=torch.float32)
stateM0 = (-1, h0, (p0, readout))
putState = composeST( backPropAt(sequence_length, updateParameterState(optimizer, f.mse_loss)) # 😱😱😱😱😱
        ,  composeST( resetHiddenStateAt(sequence_length, h0)
                    , incrementCounter))
getHiddenStates = nonAutonomousStateful(  updateHiddenState
                                        , noParamUpdate  
                                        , putState) # 😱😱😱😱😱


getOutputs = lambda initState: compose(   map(hideStateful)
                                        , drop(1)  # non-autnonmous scan returns h0 w/o +input, whose readout we don't care
                                        , getHiddenStates(initState)) 

outputs = getOutputs(stateM0)
doEpochs = mapcat(outputs)
epochs = (take(sequence_length * num_examples, cycle(zip(Xs, Ys))) for _ in range(num_epochs))

_, (pN, readoutN) = last(doEpochs(epochs))


with torch.no_grad():
    xs_test, ys_test = tee(test_loader, 2)
    xtream_test, targets_test = map(compose(lambda x: (x, None), fst), xs_test), map(snd, ys_test)

    def getReadout(pair):
        h, (_, rd) = pair 
        return rd(h)
    testOuputs = compose( map(getReadout)
                        , getOutputs((-1, h0, (pN, readoutN))))
    accuracy = compose(   curry(reduce)(add)  
                        , drop(1)
                        , take_nth(sequence_length)
                        , cons(None)
                        , map(f.mse_loss))(testOuputs(xtream_test), targets_test)
    print(accuracy)
