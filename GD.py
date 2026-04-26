import numpy as np

"""Простейший градиентный спуск для функции (поиск минимума)"""

def func(x):
    return x**2

def df(x):
    return 2 * x

N = 20
x = 2.5
lmd = 0.9

for i in range(N):
    lmd = 1/min(i+1, 100)
    x = x - lmd * np.sign(df(x))

print(x)
print(int(x))