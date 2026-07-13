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
w_1 = w_2 = np.zeros(N) # начальные нулевые значения параметров модели
n_iter = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20 # размер мини-батча (величина K = 20)

Qe_1 = Qe_2 = np.average(loss(w_1, coord_x, coord_y))# начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

# Визуализация
g = DynamicGraphs("Аппроксимация функции", (1, 2), figsize=(10, 6))
ax_1_idx, ax_2_idx = (0, 0), (0, 1)
g.set_text_axis("SDG + L2", ax_1_idx, (0.1, 0.88))
g.set_text_axis("SDG", ax_2_idx, (0.1, 0.88))
g.set_default_text_fig(n_iter, r"$model: a(x) = w_0 + w_1x + w_2x^2 + w_3x^3 + w_4x^4$")        # 0
g.set_text_axis(f"Q = {Qe_1:.2f}", ax_1_idx, to_updata=True)                                       # 1
g.set_text_axis(f"Q = {Qe_2:.2f}", ax_2_idx, to_updata=True)                                       # 2

func_str = r"$f(x) = 0.5x + 0.2x^2 - 0.05x^3 + 0.2sin(4x) - 2.5$"
g.drow_graph(coord_x, coord_y, ax_1_idx, label=func_str)
g.drow_graph(coord_x, coord_y, ax_2_idx, label=func_str)

model_str = f"$a(x) = {w_1[0]:.2f} + {w_1[1]:.2f}x + {w_1[2]:.2f}x^2 + {w_1[3]:.2f}x^3 + {w_1[4]:.2f}x^4$"
g.drow_graph(coord_x, model(w_1, coord_x), ax_1_idx, to_updata=True, color='red', label=model_str)   # 0
g.drow_graph(coord_x, model(w_1, coord_x), ax_2_idx, to_updata=True, color='red', label=model_str)   # 1

plt.pause(1)
for i in range(n_iter):
    k = np.random.randint(0, sz - batch_size-1)

    x_train = coord_x[k : k+batch_size]
    y_train = coord_y[k : k+batch_size]

    Qe_1 = lm * np.mean(loss(w_1, x_train, y_train)) + (1 - lm) * Qe_1
    mask = np.array([0, 1, 1, 1, 1])
    grad = np.mean(dL(w_1, x_train, y_train), axis=1) + lm_l2 * mask * w_1
    w_1 = w_1 - eta * grad

    new_y1 = model(w_1, coord_x)
    updata_formula = f"$a(x) = {w_1[0]:.2f} + {w_1[1]:.2f}x + {w_1[2]:.2f}x^2 + {w_1[3]:.2f}x^3 + {w_1[4]:.2f}x^4$"
    g.updata_graphs(coord_x, new_y1, 0, new_label=updata_formula)
    g.update_text(0, f'Итерация: {i + 1} / {n_iter}')
    g.update_text(1, f"Q = {Qe_1:.2f}")

    # Без L2:
    Qe_2 = lm * np.mean(loss(w_2, x_train, y_train)) + (1 - lm) * Qe_2
    grad = np.mean(dL(w_2, x_train, y_train), axis=1)
    w_2 = w_2 - eta * grad

    new_y2 = model(w_2, coord_x)
    updata_formula = f"$a(x) = {w_2[0]:.2f} + {w_2[1]:.2f}x + {w_2[2]:.2f}x^2 + {w_2[3]:.2f}x^3 + {w_2[4]:.2f}x^4$"
    g.updata_graphs(coord_x, new_y2, 1, new_label=updata_formula)
    g.update_text(2, f"Q = {Qe_2:.2f}")

    g.legend_all_ax()
    plt.pause(0.01)


Q_1 = np.average(loss(w_1, coord_x, coord_y))
Q_2 = np.average(loss(w_2, coord_x, coord_y))

print(f"w_1 = {w_1}")
print(f"Qe_1 = {Qe_1}")
print(f"Q_1 = {Q_1}", end='\n\n')
print(f"w_2 = {w_2}")
print(f"Qe_2 = {Qe_2}")
print(f"Q_2 = {Q_2}")

plt.show()