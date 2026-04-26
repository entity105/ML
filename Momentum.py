import numpy as np


def func(x):
    return -0.5 * x + 0.2 * x ** 2 - 0.01 * x ** 3 - 0.3 * np.sin(4*x)

def df(x):
    return -0.5 + 0.4*x - 0.03*x**2 - 1.2*np.cos(4*x)

eta = 0.1
x_0 = -3.5
x = x_0
N = 200
gamma = 0.8
v = 0
x_2 = x_0

for i in range(N):
    v = gamma * v + (1 - gamma) * eta * df(x)
    x = x - v

for i in range(N):
    x_2 = x_2 - eta * df(x_2)

print(x)
print(x_2)