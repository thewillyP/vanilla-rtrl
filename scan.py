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
        s_, h0, p0 = state 
        s, fn1_, fn2_ = put(s_, fn1, fn2)  # Will always have access to original fn under closure
        h1 = fn1_(h0, p0, x) # fn1 dictate behavior and we want to use state to cchange behavior
        p1 = fn2_(h1, p0, x)
        return s, h1, p1
    return recurrenceBinopStateful


nonAutonomous = curry(compose(scan, createRecurrenceBinop))  # [h_init, h1, h2, h3, ...]
nonAutonomousStateful = curry(compose(scan, createRecurrenceBinopStateful))

@curry
def rnnTransition(W_in, W_rec, b_rec, activation, alpha, h, x):
    return (1 - alpha) * h + alpha * activation(f.linear(x, W_in, bias=None) + f.linear(h, W_rec, bias=b_rec))

rnnTransition_ = curry(lambda actv, alph, win, wrec, brec, h, x: rnnTransition(win, wrec, brec, actv, alph, h, x))

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
def hideStateful(triplet):
    _, h, p = triplet
    return h, p



@curry
def incrementCounter(s, fn1, fn2):
    return (s+1, fn1, fn2)

@curry # code dup bc trying to compose 3 args at once
def composeST(st2, st1):
    def c(s, h, p):
        s1, h1, p1 = st1(s, h, p)
        return st2(s1, h1, p1)
    return c

@curry
def backPropAt(n0, parameterUpdateFn, s, fn1, fn2):
    return (s, fn1, parameterUpdateFn) if s > 0 and (s+1) % n0 == 0 else (s, fn1, fn2)  # s+1 bc backprop right before hidden state reset


tt = 0
@curry
def resetHiddenStateAt(n0, h_reset, s, fn1, fn2):
    resetFn = lambda _, p_, x_: fn1(h_reset, p_, x_)
    return (s, resetFn, fn2) if s % n0 == 0 else (s, fn1, fn2)



def epochsIO(n: int, loader: torch.utils.data.DataLoader):
    return (loader for _ in range(n))


linear_ = curry(lambda w, b, h: f.linear(h, w, bias=b))

noParamUpdate = lambda _, fp, __: fp

@curry
def updateHiddenState(h, parameters, observation):
    x, _ = observation
    forwProp, _ = parameters
    return forwProp(h, x)


step = 0

# TODO: Figure out how to make this purely functional alter. 
@curry
def updateParameterState(optimizer, lossFn, h, parameters, observation):
    global step # 😱😱😱😱😱

    _, label = observation


    if label is not None:  # None is a substitute for the Maybe monad for now
        _, readout = parameters
        loss = lossFn(readout(h), label)
        optimizer.zero_grad()  ### 😱😱😱😱
        loss.backward()  # 😱😱😱😱
        optimizer.step() # 😱😱😱😱😱

        # 😱😱😱
        if (step+1) % 100 == 0:
            print (f'Step [{step+1}], Loss: {loss.item():.10f}')
        step += 1
    return parameters  # autograd state implictly updates these guys. 


@curry
def supervisions( lossFn: Callable[[T, Y], Z]
                , outputMap: Callable[[Iterator[X]], Iterator[T]]
                , inputs: Iterator[X]
                , targets: Iterator[Y]) -> Iterator[Z]:
    return map(lossFn, outputMap(inputs), targets) 


def totalStatistic(compare: Callable[[X, Y], Z], aggregate: Callable[[Z, Z], T]):
    return compose(curry(reduce)(aggregate), supervisions(compare))


# def closure1(forwProp, h, parameters, observation):
#     x, _ = observation
#     return forwProp(parameters, h, x)



# @curry
# def updateParameterState( backProp: Callable[[torch.Tensor], tuple[torch.Tensor, ...]]
#                         , createTransition: Callable[[tuple[torch.Tensor, ...], Callable[[H, X], H]]]
#                         , createLossFn: Callable[[H, Y], torch.Tensor]
#                         , h: H
#                         , parameters
#                         , observation: tuple[X, Y]):
#     _, label = observation
#     _, lossFn = parameters  # lossFn = loss .  readout
#     loss = lossFn(h, label)
#     parameterized: tuple[torch.Tensor, ...] = backProp(loss)
#     return createTransition(parameterized), createLossFn(parameterized)

#     # return createTransition(W_in_, W_rec_, b_rec), createLossFn(W_out, b_out)
#     #rnnTransition' = rnnTransition_(act, alph)  <-- comes from user defined. 
#     pass  # TODO, returns rnnTransition'(win', wrec', wbin')




# @curry 
# def temp(updateForw, h, parameters, observation):
#     return updateForw(h, parameters, observation)

# def backProp(h, parameters, observation):
#     _, label = observation
#     forwardProp = parameters
#     torch.autograd.grad()


# optimizer.zero_grad()
# l.backward()
# optimizer.step()

# getHiddenStates = curry(nonAutonomous(updateHiddenState, noParamUpdate))

# getHiddenStatesStateful = lambda updateState: nonAutonomousStateful(  updateHiddenState
#                                                                     , noParamUpdate
#                                                                     , updateState)








# def backPropagateIO(loss, optimizer):
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()




"""
parameter should be a function that takes a previous hidden state and returns a new hidden state.
Therefore, parameter ought to be the forward propagate. 
OHO: forw= NN forw + backprop. 
Backprop needs: loss, label, nn output, current theta, optimizer needs to be here to combine prev theta with new -> new theta. 
forw needs: x, current theta -> nn output. 

But what if NN is an RNN? 
For the forw output to go through, I need both x, current theta, AND prev hidden state.
This is only true IF we dont reset RNN hidden state after each training iteration. 
Question is, should we pass RNN hidden state from oe training session to the next?
Sure, we should allow. Therefore the forward pass outputs a tuple: (theta_i+1, hidden_i+1)

Update: is it static? Yes, the update rules for updating the forward pass do not change since they are parameterized by meta learning rate.
Given a closure, it takes the current forward pass, the readout which is not a function of x, but of x_val

observation = (x_train, y_train, x_val, y_val)
fp = parameter = factory(hidden_i-1, loss, x_train, y_train) -> hidden_i
h = hidden_i-1 

observation = (x_train, y_train)
fp = parameter = rnn(hidden_i-1, x_train) - > hidden_i

OHO: fp is a factory
RNN: fp is just rnn forward. 

updateParameter: stuff -> factory
updateParameter: stuff -> rnn(...)

Vanila:
forw: needs x, curr hidden state -> output
backward: 


OHO needs to nest two recurrence sequence. One for the forward where forward does the whole rnn training sequence on x number batches.

OHO readout is loss perf on validation data. That is our loss.
Vanilla readout is just linear readout
Both readout and transition are updated since both are functions of parameter.
OHO: readout is not a function of hyperparameters. Readout just takes the hidden state and computes a loss on them.
This is not require using hyperparameters a.k.a weights analogy.

But in Van, the readout is a function of the weights a.k.k hyperparaneters. 

So in OHO, dont expect readout fn to change, but in rnn, do change them. 

The only thing readout needs in either is the previous hidden state and the labels. 
So no need for the input


"""



"""S
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
