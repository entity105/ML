import numpy as np
from graphs import DynamicGraphs
from matplotlib import pyplot as plt


def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


def model(w, x):
    return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3


coord_x = np.arange(-4.0, 6.0, 0.1)

x_train = np.array([[_x**i for i in range(4)] for _x in coord_x]) # обучающая выборка - матрица (n x 4)
y_train = func(coord_x) # целевые выходные значения
w = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train
model_res = model(w, coord_x)
Q = np.average((model_res - y_train)**2)

g = DynamicGraphs("МНК: Аппроксимация функции", figsize=(10, 6))
g.set_text_axis(f"Q = {Q:.4f}", coord_text=(0.03, 0.95))
g.set_text_axis(f"w = {[round(float(w_i), 2) for w_i in w]}", coord_text=(0.03, 0.85))

func_str = r"$f(x) = 0.5x + 0.2x^2 - 0.05x^3 + 0.2sin(4x) - 2.5$"
g.draw_graph(coord_x, y_train, label=func_str)

model_str = r"$a(x) = w_0 + w_1x + w_2x^2 + w_3x^3$"
g.draw_graph(coord_x, model_res, color='red', label=model_str)

plt.show()

print(f"w = {w}")