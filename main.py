# %%
import torch 
import numpy as np
from core import RNN
from functions.Function import Function
from functions import *
from gen_data.Add_Memory_Task import *
import matplotlib.pyplot as plt

t1: float = 4
t2: float = 6
a: float = 2
b: float = -2
t1_dur: float = 0.99
t2_dur: float = 0.99
st, et = 0, 10
domain = range(st, et)
addMemoryTask = createAddMemoryTask(t1, t2, a, b, t1_dur, t2_dur)
discretizedStates = map(addMemoryTask, domain)

# %% 
plt.figure()
plt.plot(np.arange(0, 15, 1000), list(map(addMemoryTask, np.arange(0, 15, 1000))))
plt.show()

# W_in = np.eye(2)
# W_rec = np.eye(2)
# W_out = np.eye(2)
# b_rec = np.zeros(2)
# b_out = np.zeros(2)

# rnn: RNN = RNN(W_in
#         , W_rec
#         , W_out
#         , b_rec
#         , b_out
#         , activation=identity
#         , alpha=1
#         , output=softmax
#         , loss=softmax_cross_entropy)



# class Test_Exact_Learning_Algorithms(unittest.TestCase):
#     """Verifies that BPTT algorithms gives same aggregate weight change as
#     RTRL for a very small learning rate, while also checking that the
#     recurrent weights did change some amount (i.e. learning rate not *too*
#     small that this is trivially true)."""

#     @classmethod
#     def setUpClass(cls):

#         cls.task = Add_Task(4, 6, deterministic=True, tau_task=1)
#         cls.data = cls.task.gen_data(400, 0)

#         n_in = cls.task.n_in
#         n_h = 16
#         n_out = cls.task.n_out

#         cls.W_in = np.random.normal(0, np.sqrt(1/(n_in)), (n_h, n_in))
#         M_rand = np.random.normal(0, 1, (n_h, n_h))
#         cls.W_rec = np.linalg.qr(M_rand)[0]
#         cls.W_out = np.random.normal(0, np.sqrt(1/(n_h)), (n_out, n_h))
#         cls.W_FB = np.random.normal(0, np.sqrt(1/n_out), (n_out, n_h))

#         cls.b_rec = np.zeros(n_h)
#         cls.b_out = np.zeros(n_out)

#     def test_small_lr_case(self):

#             alpha = 1

#             self.rnn_1 = RNN(self.W_in, self.W_rec, self.W_out,
#                              self.b_rec, self.b_out,
#                              activation=tanh,
#                              alpha=alpha,
#                              output=softmax,
#                              loss=softmax_cross_entropy)

#             self.rnn_2 = RNN(self.W_in, self.W_rec, self.W_out,
#                              self.b_rec, self.b_out,
#                              activation=tanh,
#                              alpha=alpha,
#                              output=softmax,
#                              loss=softmax_cross_entropy)

#             self.rnn_3 = RNN(self.W_in, self.W_rec, self.W_out,
#                              self.b_rec, self.b_out,
#                              activation=tanh,
#                              alpha=alpha,
#                              output=softmax,
#                              loss=softmax_cross_entropy)

#             lr = 0.00001
#             self.optimizer_1 = Stochastic_Gradient_Descent(lr=lr)
#             self.learn_alg_1 = RTRL(self.rnn_1)
#             self.optimizer_2 = Stochastic_Gradient_Descent(lr=lr)
#             self.learn_alg_2 = Future_BPTT(self.rnn_2, 25)
#             self.optimizer_3 = Stochastic_Gradient_Descent(lr=lr)
#             self.learn_alg_3 = Efficient_BPTT(self.rnn_3, 100)

#             monitors = []

#             np.random.seed(1)
#             self.sim_1 = Simulation(self.rnn_1)
#             self.sim_1.run(self.data, learn_alg=self.learn_alg_1,
#                            optimizer=self.optimizer_1,
#                            monitors=monitors,
#                            verbose=False)

#             np.random.seed(1)
#             self.sim_2 = Simulation(self.rnn_2)
#             self.sim_2.run(self.data, learn_alg=self.learn_alg_2,
#                            optimizer=self.optimizer_2,
#                            monitors=monitors,
#                            verbose=False)

#             np.random.seed(1)
#             self.sim_3 = Simulation(self.rnn_3)
#             self.sim_3.run(self.data, learn_alg=self.learn_alg_3,
#                            optimizer=self.optimizer_3,
#                            monitors=monitors,
#                            verbose=False)

#             #Assert networks learned similar weights with a small tolerance.
#             assert_allclose(self.rnn_1.W_rec, self.rnn_2.W_rec, atol=1e-4)
#             assert_allclose(self.rnn_2.W_rec, self.rnn_3.W_rec, atol=1e-4)
#             #But that there was some difference from initialization
#             self.assertFalse(np.isclose(self.rnn_1.W_rec,
#                                         self.W_rec, atol=1e-4).all())
