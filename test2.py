from memory_profiler import profile


from itertools import tee

from func import *
from toolz.curried import accumulate, last, compose, take, map



# @profile
def func():
    n = 100000000000
    # test = ((x, x) for x in range(n))
    # # test = list(test)
    # xs, ys = tee(test)
    # i1 = map(fst, xs)
    # i2 = map(snd, ys)
    # x = list(map2(lambda x, y: x, i1, i2))

    def addOne(x):
        print('1')
        return x + 1
    
    def addOnePrime(x):
        print('2')
        return x + 1

    ys = range(n)
    x = compose(take(2), map(addOnePrime), map(addOne))(ys)
    # x = take(2, map(lambda x: x + 2, map(addOne, map(addOne, ys))))
    x = list(x)
    print(x)
    # x =last(scan(lambda x, y: x + y, 0, map2(lambda x, y: x + y, i1, i2)))
    # x = sum(i1) + sum(i2)


def map2(f1, i1, i2):
    for pair in zip(i1, i2):
        x, y = pair 
        yield f1(x, y)

if __name__ == '__main__':
    func()