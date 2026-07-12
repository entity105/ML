import numpy as np
from graphs import DynamicGraphs
from matplotlib import pyplot as plt


# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5


# модель
def model(w, x):
    xv = np.array([x ** n for n in range(len(w))])      # 1 x 5
    # print(xv.shape)
    return w @ xv


# функция потерь
def loss(w, x, y):
    return (model(w, x) - y) ** 2


# производная функции потерь
def dL(w, x, y):
    xv = np.array([x ** n for n in range(len(w))])  # 5 x n
    return 2 * (model(w, x) - y) * xv


coord_x = np.arange(-4.0, 6.0, 0.1) # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

N = 5 # сложность модели (полином степени N-1)
lm_l2 = 2 # коэффициент лямбда для L2-регуляризатора
sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.zeros(N) # начальные нулевые значения параметров модели
n_iter = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20 # размер мини-батча (величина K = 20)

Qe = np.average(loss(w, coord_x, coord_y))# начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

# Визуализация
g = DynamicGraphs("Аппроксимация функции", figsize=(10, 6))
g.set_default_text_fig(n_iter, r"$model: a(x) = w_0 + w_1x + w_2x^2 + w_3x^3 + w_4x^4$")
g.set_text_axis(f"Q = {Qe:.2f}", to_updata=True)

func_str = r"$f(x) = 0.5x + 0.2x^2 - 0.05x^3 + 0.2sin(4x) - 2.5$"
g.drow_graph(coord_x, coord_y, base_setting=False, color='blue', linewidth=3, label=func_str)

model_str = f"$a(x) = {w[0]:.2f} + {w[1]:.2f}x + {w[2]:.2f}x^2 + {w[3]:.2f}x^3 + {w[4]:.2f}x^4$"
g.drow_graph(coord_x, model(w, coord_x), to_updata=True, base_setting=False, color='red', linewidth=3, label=model_str)

plt.pause(1)
for i in range(n_iter):
    k = np.random.randint(0, sz - batch_size-1)

    x_train = coord_x[k : k+batch_size]
    y_train = coord_y[k : k+batch_size]

    Qe = lm * np.mean(loss(w, x_train, y_train)) + (1 - lm) * Qe
    mask = np.array([0, 1, 1, 1, 1])
    grad = np.mean(dL(w, x_train, y_train), axis=1) + lm_l2 * mask * w
    w = w - eta * grad

    new_y = model(w, coord_x)
    updata_formula = f"$a(x) = {w[0]:.2f} + {w[1]:.2f}x + {w[2]:.2f}x^2 + {w[3]:.2f}x^3 + {w[4]:.2f}x^4$"
    g.updata_graphs(coord_x, new_y, new_label=updata_formula)
    g.update_text(0, f'Итерация: {i + 1} / {n_iter}')
    g.update_text(1, f"Q = {Qe:.2f}")

    plt.pause(0.1)


Q = np.average(loss(w, coord_x, coord_y))

print(w)
print(Qe)
print(Q)

plt.show()