import matplotlib.pyplot as plt
import numpy as np
import graphs as g


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
fig, ax1, ax2 = g.create_window(1, 2, figsize=(16, 7.5), title='Градиентный спуск')
fig.subplots_adjust(top=0.88, bottom=0.1)
itr_str = fig.text(0.2, 0.90, f'Итерация: 0 / {N}', ha='center', fontsize=14)
fig.text(0.4, 0.90, formula, fontsize=14, fontweight='bold')

g.drow2Dgraph(coord_x, coord_y, ax1)
g.drow2Dgraph(coord_x, coord_y, ax2)

text1 = g.add_text(ax1, f'Координата: ({x_0:.2f}, {func(x_0):.2f})')
text2 = g.add_text(ax2, f'Координата: ({x_0:.2f}, {func(x_0):.2f})')
g.add_text(ax1, r'$v_{n+1} = \gamma \cdot v_n + (1 - \gamma )\eta \cdot \nabla f(x)$' '\n' r'$x_{n+1} = x_n - v_{n+1}$', x=0.4)
g.add_text(ax2, r'$x_{n+1} = x_n - \eta \cdot \nabla f(x)$', x=0.6)
point, = ax1.plot([x_0], [func(x_0)], 'ro', markersize=12)
point2, = ax2.plot([x_0], [func(x_0)], 'ro', markersize=12)

plt.pause(2)
for i in range(N):
    # С оптимизатором
    v = gamma * v + (1 - gamma) * eta * df(x_1)
    x_1 = x_1 - v

    # Чистый GD
    x_2 = x_2 - eta * df(x_2)

    itr_str.set_text(f'Итерация: {i + 1} / {N}')
    y_1 = func(x_1)
    point.set_data([x_1], [y_1])
    text1.set_text(f'Координата: ({x_1:.2f}, {y_1:.2f})')

    y_2 = func(x_2)
    point2.set_data([x_2], [y_2])
    text2.set_text(f'Координата: ({x_2:.2f}, {y_2:.2f})')

    plt.pause(0.1)

print(x_1)
print(x_2)

plt.show()