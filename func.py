from typing import TypeVar, Callable, Generic, Generator, Iterator
from functools import reduce
from toolz import curry

T = TypeVar('T') 
X = TypeVar('X')

@curry
def scan(f: Callable[[T, X], T], state: T, it: Iterator[X]) -> Generator[T, None, None]:
    yield state
    for x in it:
        state = f(state, x)
        yield state