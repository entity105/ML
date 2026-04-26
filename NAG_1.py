import numpy as np


def func(x):
    return 0.4 * x + 0.1 * np.sin(2*x) + 0.2 * np.cos(3*x)

def df(x):
    return 0.4 + 0.2 * np.cos(2*x) - 0.6 * np.sin(3*x)

n = 1.0
x_0 = 4.0
x = x_0
N = 500
gamma = 0.7
v = 0

for i in range(500):
    v = gamma * v + (1 - gamma) * n * df(x - gamma * v)
    x = x - v
print(x)