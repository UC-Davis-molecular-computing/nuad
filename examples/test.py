from typing import Callable


def func() -> bool:
    return True


f1: Callable[[], bool] = func  # okay
f2: Callable[[], float] = func  # error: f2's return type does not match func's

r1: float = f1()  # error: r1's type float does not match f1's return type bool
r2: float = func()  # error: r2's type float does not match func's return type bool
r3: float = f2()  # okay, given how f2 is declared
