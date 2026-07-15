import numpy as np
from graphs import MovePoint
from matplotlib import pyplot as plt


def func(x):
    return 2 * x + 0.1 * x ** 3 + 2 * np.cos(3*x)

def df(x):
    return 2 + 0.3 * x**2 - 6 * np.sin(3*x)

n = 0.5
x_0 = 4.0
x_1 = x_2 = x_3 = x_4 = x_0
N = 50
alpha = 0.8
G = 0
epsilon = 0.01

coord_x = np.arange(-3, 8, 0.1)
coord_y = func(coord_x)

w = MovePoint('GD: оптимизатор RMSProp', (1, 2), figsize=(16, 7.5))
w.set_default_text_fig(N, '$f(x) = 2x + 0.1x^3 + 2cos(3x)$')    # 0
w.draw_graph(coord_x, coord_y, 0)
w.draw_graph(coord_x, coord_y, 1)

w.set_text_axis("GD + RMSProp", 0, xy:=(0.4, 0.95))
w.set_text_axis("GD", 1, xy)

y_0 = func(x_0)
w.set_text_axis(f"Координата: ({x_0:.2f}, {y_0:.2f})", 0, xy:=(0.03, 0.95), to_update=True) # 0
w.set_text_axis(f"Координата: ({x_0:.2f}, {y_0:.2f})", 1, xy, to_update=True) # 1

w.make_point(x_0, y_0, 0, to_update=True) # 0
w.make_point(x_0, y_0, 1, to_update=True) # 1

plt.pause(1)
for i in range(N):
    G = alpha*G + (1-alpha)*df(x_1)**2
    x_1 = x_1 - n*df(x_1)/(G**0.5 + epsilon)

    lmd = 1 / min(i + 1, 100)
    x_2 = x_2 - lmd * np.sign(df(x_2))

    w.update_text(0, f"Итерация: {i + 1} / {N}")
    y_1, y_2 = func(x_1), func(x_2)
    w.update_text(1, f"Координата: ({x_1:.2f}, {y_1:.2f})")
    w.update_text(2, f"Координата: ({x_2:.2f}, {y_2:.2f})")

    w.update_point(x_1, y_1, 0)
    w.update_point(x_2, y_2, 1)

    plt.pause(0.1)

print(x_1, x_2, sep='\n')
plt.show()