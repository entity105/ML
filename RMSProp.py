import numpy as np


def func(x):
    return 2 * x + 0.1 * x ** 3 + 2 * np.cos(3*x)

def df(x):
    return 2 + 0.3 * x**2 - 6 * np.sin(3*x)

n = 0.5
x_0 = 4.0
x = x_0
N = 200
alpha = 0.8
G = 0
epsilon = 0.01

for i in range(N):
    G = alpha*G + (1-alpha)*df(x)**2
    x = x - n*df(x)/(G**0.5 + epsilon)
print(x)