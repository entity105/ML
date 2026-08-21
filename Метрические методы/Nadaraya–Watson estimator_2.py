import numpy as np


def func(x):
    return 0.1 * x - np.cos(x/2) + 0.4 * np.sin(3*x) + 5


np.random.seed(0)

x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
y = func(x) + np.random.normal(0, 0.2, len(x)) # значения функции по оси ординат
h = 0.5

y_est = []

for x_i in x:
    r = np.abs(x_i - x) / h
    K = 1/(np.sqrt(2*np.pi)) * np.exp(-r**2 / 2)

    res = np.sum(y * K) / np.sum(K)
    y_est.append(res)

Q = np.mean((y_est - y)**2)

# print(abs(x.reshape(-1, 1) - x) / 0.5)
# print()
# print(x.reshape(-1, 1) - x)