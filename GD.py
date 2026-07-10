import numpy as np
from matplotlib import pyplot as plt
from graphs import MovePoint

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
w = MovePoint('Градиентный спуск')
w.drow_graph(coord_x, coord_y)
w.set_text_axis(f'Итерация i = 0 / {N}, координата: ({x:.4f}, {func(x):.4f})', to_updata=True)
w.make_point(x, func(x), to_update=True)

plt.pause(1)
for i in range(N):

    lmd = 1/min(i+1, 100)
    x = x - lmd * np.sign(df(x))

    y = func(x)
    w.updata_point(x, y, 0)
    w.updata_ax_text(0, f'Итерация i = {i+1} / {N}, координата: ({x:.4f}, {y:.4f})')
    plt.pause(0.1)

print(x)
print(int(x))

plt.show()
