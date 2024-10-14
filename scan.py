from func import fst, snd, scan, reduce_, swap, uncurry
import torch 
from torch.nn import functional as f
import numpy as np
from functions import *
from gen_data.Add_Memory_Task import *
from typing import TypeVar, Callable, Generic, Generator, Iterator
from toolz.curried import curry, compose, identity, take_nth, accumulate, apply, map, concat, take, drop
from functools import reduce
from itertools import tee


T = TypeVar('T')  # T can be any type
H = TypeVar('H')
P = TypeVar('P')
X = TypeVar('X')
Y = TypeVar('Y')
Z = TypeVar('Z')
S = TypeVar('S')



@curry
def createRecurrenceBinop(fn1: Callable[[H, P, X], H], fn2: Callable[[H, P, X], P]) -> Callable[[tuple[H, P], X], tuple[H, P]]:
    @curry
    def recurrenceBinop(state: tuple[H, P], x: X) -> tuple[H, P]:
        h0, p0 = state 
        h1 = fn1(h0, p0, x)
        p1 = fn2(h1, p0, x)
        return h1, p1
    return recurrenceBinop

# don't want to crack out State Monad in python so resort to code duplication. Also don't want to have to make all my transition functions have to thread state which is unclean
@curry
def createRecurrenceBinopStateful(fn1: Callable[[H, P, X], H], fn2: Callable[[H, P], P], put: Callable[[S, H, P], tuple[S, H, P]]) -> Callable[[tuple[S, H, P], X], tuple[S, H, P]]:
    @curry
    def recurrenceBinopStateful(state: tuple[S, H, P], x: X) -> tuple[S, H, P]:
        s_, h0_, p0_ = state 
        s, h0, p0 = put(s_, h0_, p0_)
        h1 = fn1(h0, p0, x)
        p1 = fn2(h1, p0, x)
        return s, h1, p1
    return recurrenceBinopStateful


nonAutonomous = curry(compose(scan, createRecurrenceBinop))  # [h_init, h1, h2, h3, ...]
nonAutonomousStateful = curry(compose(scan, createRecurrenceBinopStateful))

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


@curry
def supervisions( lossFn: Callable[[T, Y], Z]
                , outputMap: Callable[[Iterator[X]], Iterator[T]]
                , inputs: Iterator[X]
                , targets: Iterator[Y]) -> Iterator[Z]:
    return map(lossFn, outputMap(inputs), targets) 


@curry
def hideStateful(triplet):
    _, h, p = triplet
    return h, p

@curry
def resetHiddenStateAt(n0, h_reset, s, h, p):
    return (1, h_reset, p) if n0 == s else (s+1, h, p)



linear_ = curry(lambda w, b, h: f.linear(h, w, bias=b))

# updateHiddenState = lambda h, fp, x: fp(h, x)

noParamUpdate = lambda _, fp, __: fp

def updateHiddenState(h, parameters, observation):
    x, _ = observation
    forwProp, _ = parameters
    return forwProp(h, x)

def backProp(h, parameters, observation):
    _, label = observation
    _, lossFn = parameters
    lossFn(h, label)
    pass  # TODO

# optimizer.zero_grad()
# l.backward()
# optimizer.step()

getHiddenStates = curry(nonAutonomous(updateHiddenState, noParamUpdate))

getHiddenStatesStateful = lambda updateState: nonAutonomousStateful(  updateHiddenState
                                                                    , noParamUpdate
                                                                    , updateState)


def separateLabelsIO(loader):
    as_, bs_ = tee(loader, 2)
    return as_, map(snd, bs_)

def epochsIO(n: int, loader):
    return (separateLabelsIO(loader) for _ in range(n))


def totalStatistic(compare: Callable[[X, Y], Z], aggregate: Callable[[Z, Z], T]):
    return compose(curry(reduce)(aggregate), supervisions(compare))


# def backPropagateIO(loss, optimizer):
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()





"""
I still need to pass in parameters as functions to updateHiddenState because if my backprop updates my functions,
the hidden transition function should update as well. So I can't preset what the forward pass should be. 
"""


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
