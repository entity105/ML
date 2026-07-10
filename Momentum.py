import matplotlib.pyplot as plt
import numpy as np
from graphs import MovePoint


def func(x):
    return -0.5 * x + 0.2 * x ** 2 - 0.01 * x ** 3 - 0.3 * np.sin(4*x)

def df(x):
    return -0.5 + 0.4*x - 0.03*x**2 - 1.2*np.cos(4*x)

eta = 0.1
x_0 = -3.5
x_1 = x_2 = x_0
N = 200
gamma = 0.8
v = 0

coord_x = np.arange(-5, 5, 0.1)
coord_y = func(coord_x)

formula = r'$f(x) = -0.5 x + 0.2 x^2 - 0.01 x^3 - 0.3 sin(4x)$'
ax_1_idx, ax_2_idx = (0, 0), (0, 1)

# Создаём окно, подписываем, делаем оси с графиками
w = MovePoint('Градиентный спуск', (1, 2), figsize=(16, 7.5))
w.set_default_text_fig(N, formula)
w.drow_graph(coord_x, coord_y, ax_1_idx)
w.drow_graph(coord_x, coord_y, ax_2_idx)

# Наполняем оси текстом
text_point = f'Координата: ({x_0:.2f}, {func(x_0):.2f})'
w.set_text_axis(text_point, ax_1_idx, to_updata=True)
w.set_text_axis(text_point, ax_2_idx, to_updata=True)

# Статичный текст
text_1 = r'$v_{n+1} = \gamma \cdot v_n + (1 - \gamma )\eta \cdot \nabla f(x)$' '\n' r'$x_{n+1} = x_n - v_{n+1}$'
text_2 = r'$x_{n+1} = x_n - \eta \cdot \nabla f(x)$'
w.set_text_axis(text_1, ax_1_idx, coord_text=(0.5, 0.95))
w.set_text_axis(text_2, ax_2_idx,coord_text=(0.5, 0.95))

# Делаем точки
w.make_point(x_1, func(x_1), ax_1_idx, to_update=True)
w.make_point(x_2, func(x_2), ax_2_idx, to_update=True)

plt.pause(2)
for i in range(N):
    # С оптимизатором
    v = gamma * v + (1 - gamma) * eta * df(x_1)
    x_1 = x_1 - v

    # Чистый GD
    x_2 = x_2 - eta * df(x_2)

    # Обновление
    w.update_fig_text(0, f'Итерация: {i+1} / {N}')
    w.updata_point(x_1, func(x_1), 0)
    w.updata_ax_text(0, f'Координата: ({x_1:.2f}, {func(x_1):.2f})')

    w.updata_point(x_2, func(x_2), 1)
    w.updata_ax_text(1, f'Координата: ({x_2:.2f}, {func(x_2):.2f})')

    plt.pause(0.1)

print(x_1)
print(x_2)

plt.show()