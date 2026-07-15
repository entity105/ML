import matplotlib.pyplot as plt
import numpy as np
from graphs import MovePoint


def func(x):
    return 0.4 * x + 0.1 * np.sin(2*x) + 0.2 * np.cos(3*x)

def df(x):
    return 0.4 + 0.2 * np.cos(2*x) - 0.6 * np.sin(3*x)

n = 1.0
x_0 = 4.0
x_1 = x_2 = x_0
N = 100
gamma = 0.8
v = 0

coord_x = np.arange(-3, 8, 0.1)
coord_y = func(coord_x)

w = MovePoint('GD: оптимизатор NAG', (1, 2), figsize=(16, 7.5))
ax_1_idx, ax_2_idx = (0, 0), (0, 1)
w.set_default_text_fig(N, '$f(x) = 0.4 x + 0.1 sin(2x) + 0.2 cos(3x)$')    # 0
w.drow_graph(coord_x, coord_y, ax_1_idx)
w.drow_graph(coord_x, coord_y, ax_2_idx)

w.set_text_axis("GD + NAG", ax_1_idx, (0.4, 0.95))
w.set_text_axis("GD", ax_2_idx, (0.4, 0.95))
w.set_text_axis(f"Координата: ({x_0:.2f}, {func(x_0):.2f})", ax_1_idx, (0.03, 0.95), to_updata=True) # 0
w.set_text_axis(f"Координата: ({x_0:.2f}, {func(x_0):.2f})", ax_2_idx, (0.03, 0.95), to_updata=True) # 1

w.make_point(x_1, func(x_1), ax_1_idx, to_update=True) # 0
w.make_point(x_2, func(x_2), ax_2_idx, to_update=True) # 1

plt.pause(1)
for i in range(N):
    v = gamma * v + (1 - gamma) * n * df(x_1 - gamma * v)
    x_1 = x_1 - v

    lmd = 1 / min(i + 1, 100)
    x_2 = x_2 - lmd * np.sign(df(x_2))

    w.update_text(0, f"Итерация: {i + 1} / {N}")
    y_1, y_2 = func(x_1), func(x_2)
    w.update_text(0, f"Координата: ({x_1:.2f}, {y_1:.2f})")
    w.update_text(1, f"Координата: ({x_2:.2f}, {y_2:.2f})")

    w.update_point(x_1, y_1, 0)
    w.update_point(x_2, y_2, 1)

    plt.pause(0.2)

print(x_1)
print(x_2)

plt.show()