import numpy as np
from matplotlib import pyplot as plt

import graphs as g

"""Простейший градиентный спуск для функции (поиск минимума)"""

def func(x):
    return x**2

def df(x):
    return 2 * x

N = 50
x = 4.5
lmd = 0.9


coord_x = np.arange(-5, 5, 0.1)
coord_y = func(coord_x)

# График
fig, ax = g.create_window(title='Градиентный спуск')
g.drow2Dgraph(coord_x, coord_y, ax, linewidth=2,
              name_func='$y = x^2$')

point, = ax.plot([], [], 'ro', markersize=12, label='Current point')


for i in range(N):
    y = func(x)
    point.set_data([x], [y])
    ax.set_title(f'Градиентный спуск, итерация i = {i+1}, координата: ({x:.4f}, {y:.4f})')
    plt.pause(0.1)

    lmd = 1/min(i+1, 100)
    x = x - lmd * np.sign(df(x))

print(x)
print(int(x))

plt.show()

# drow2Dgraph(coord_x, coord_y)