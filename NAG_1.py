import matplotlib.pyplot as plt
import numpy as np
import graphs as g


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

coord_x = np.arange(0, 8, 0.1)
coord_y = func(coord_x)

fig, ax_1, ax_2 = g.create_window(1, 2, figsize=(16, 7.5), title='Градиентный спуск')
g.drow2Dgraph(ax_1, coord_x, coord_y)
g.drow2Dgraph(ax_2, coord_x, coord_y)



for i in range(500):
    v = gamma * v + (1 - gamma) * n * df(x - gamma * v)
    x = x - v
print(x)

plt.show()