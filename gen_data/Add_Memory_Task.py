import numpy as np
from gen_data.Task import Task
from typing import Callable
from functools import reduce


def apply(x, f):
    return f(x)

def compose(*callables):
    return lambda x: reduce(apply, callables, x)

def createUnitSignal(startTime: float, duration: float) -> Callable[[float], float]:
    def test(t):
        return 1.0 * float(0 <= t - startTime <= duration) 
    return test
    # return lambda t: 1.0 * (0 <= t - startTime <= duration) 

"""
    y(t) = x(t - t_1) + x(t - t_2)           (1)
"""
# :: time (<No prev state bc stateless>, Action) -> (x1, x2, y) (State)
def createAddMemoryTask(  t1: float
                        , t2: float
                        , a: float
                        , b: float
                        , t1_dur: float
                        , t2_dur: float
                        , outT: float) -> Callable[[float], tuple[float, float, float]]:
    x1 = compose(createUnitSignal(outT - t1, t1_dur), lambda x: a*x)
    x2 = compose(createUnitSignal(outT - t2, t2_dur), lambda x: b*x)
    y = lambda t: x1(t - t1) + x2(t - t2)
    return lambda t: (x1(t), x2(t), y(t))







# class Add_Memory_Task(Task):
#     """

#     y(t) = x(t - t_1) + x(t - t_2)           (1)

#     The inputs and outputs each have a redundant dimension representing the
#     complement of the outcome (i.e. x_1 = 1 - x_0), because keeping all
#     dimensions above 1 makes python broadcasting rules easier."""

#     def __init__(self, t1, t2, t1_duration, t2_duratio):
#         """Initializes an instance of this task by specifying the temporal
#         distance of the dependencies, whether to use deterministic labels, and
#         the timescale of the changes.

#         Args:
#             t_1 (int): Number of time steps for first dependency
#             t_2 (int): Number of time steps for second dependency
#             deterministic (bool): Indicates whether to take the labels as
#                 the exact numbers in Eq. (1) OR to use those numbers as
#                 probabilities in Bernoulli outcomes.
#             tau_task (int): Factor by which we temporally 'stretch' the task.
#                 For example, if tau_task = 3, each input (and label) is repeated
#                 for 3 time steps before being replaced by a new random
#                 sample."""

#         #Initialize a parent Task object with 2 input and 2 output dimensions.
#         super().__init__(2, 2)

#         #Dependencies in coin task
#         self.t_1 = t_1
#         self.t_2 = t_2


#     def gen_dataset(self, N):
#         """Generates a dataset according to Eq. (1)."""

        
#         y = 0.5 + 0.5 * np.roll(x, self.t_1) - 0.25 * np.roll(x, self.t_2)
#         if not self.deterministic:
#             y = np.random.binomial(1, y, N)
#         X = np.array([x, 1 - x]).T
#         Y = np.array([y, 1 - y]).T

#         #Temporally stretch according to the desire timescale of change.
#         X = np.tile(X, self.tau_task).reshape((self.tau_task*N, 2))
#         Y = np.tile(Y, self.tau_task).reshape((self.tau_task*N, 2))

#         return X, Y, None, None, None