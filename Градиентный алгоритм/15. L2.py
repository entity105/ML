import numpy as np
from graphs import DynamicGraphs
from matplotlib import pyplot as plt


def func(x):
    return 0.1 * x + 0.1 * x ** 2 - 0.5 * np.sin(2*x) + 1 * np.cos(4*x) + 10

def model(w, x):
    return x @ w   # Выдаёт вектор из значений

x = np.arange(-3.0, 4.1, 0.1) # значения по оси абсцисс (Ox) с шагом 0,1
y = np.array(func(x)) # значения функции по оси ординат

N = 22  # размер признакового пространства (степень полинома N_bt-1)
lm = 20  # параметр лямбда для L2-регуляризатора

X = np.array([[a ** n for n in range(N)] for a in x])  # матрица входных векторов
IL = lm * np.eye(N)  # матрица lambda*I
IL[0][0] = 0  # первый коэффициент не регуляризуем

X_train = X[::2]  # обучающая выборка (входы)
Y_train = y[::2]  # обучающая выборка (целевые значения)

w = np.linalg.inv(X_train.T @ X_train + IL) @ X_train.T @ Y_train
model_res = model(w, X)
Q = np.average((model_res - y)**2)

g = DynamicGraphs("L2: Аппроксимация функции", figsize=(12, 6))
g.set_text_axis(f"Q = {Q:.4f}", coord_text=(0.03, 0.95))
g.set_text_axis(f"w = {[round(float(w_i), 2) for w_i in w]}", coord_text=(0.03, 0.85))

func_str = r"$f(x) = 0.1x + 0.1x^2 - 0.5sin(2x) + cos(4x) + 10$"
g.draw_graph(x, y, label=func_str)

model_str = r"$a(x) = w_0 + w_1x + w_2x^2 + w_3x^3 + ... + w_{21}x^{21}$"
g.draw_graph(x, model_res, color='red', label=model_str)
g.update_legend()

plt.show()

print(w)

print(Q)