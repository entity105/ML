import numpy as np
import matplotlib.pyplot as plt
from graphs import DynamicGraphs

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return -0.5 * x ** 2 + 0.1 * x ** 3 + np.cos(3 * x) + 7


# модель
def model(w, x):
    xv = np.array([x ** n for n in range(len(w))])  # w_0 + w_1*x + w_2*x^2 + w_3*x^3 + w_4*x^4
    return w.T @ xv


# функция потерь
def loss(w, x, y):
    return (model(w, x) - y) ** 2


# производная функции потерь
def dL(w, x, y):
    xv = np.array([x ** n for n in range(len(w))])
    return 2 * (model(w, x) - y) * xv


coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

N = 5 # сложность модели (полином степени N-1)
lm_l1 = 2.0 # коэффициент лямбда для L1-регуляризатора
sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001, 0.000002]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w_1 = w_2 = np.zeros(N) # начальные нулевые значения параметров модели
n_iter = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20 # размер мини-батча (величина K = 20)

Qe_1 = Qe_2 = np.average(loss(w_1, coord_x, coord_y))   # начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

# Визуализация
g = DynamicGraphs("Аппроксимация функции", (1, 2), figsize=(10, 6))
g.set_default_text_fig(n_iter, r"$model: a(x) = w_0 + w_1x + w_2x^2 + w_3x^3 + w_4x^4$") # 0
g.set_text_axis("SDG + L1", 0, (0.1, 0.88))
g.set_text_axis("SDG", 1, (0.1, 0.88))
g.set_text_axis(f"Q = {Qe_1:.2f}", 0, to_update=True)    # 1
g.set_text_axis(f"Q = {Qe_2:.2f}", 1, to_update=True)    # 2

func_str = r"$f(x) = -0.5x^2 + 0.1x^3 + cos(3x) + 7$"
g.draw_graph(coord_x, coord_y, 0, label=func_str)
g.draw_graph(coord_x, coord_y, 1, label=func_str)

model_str = f"$a(x) = {w_1[0]:.2f} + {w_1[1]:.2f}x + {w_1[2]:.2f}x^2 + {w_1[3]:.2f}x^3 + {w_1[4]:.2f}x^4$"
g.draw_graph(coord_x, model(w_1, coord_x), 0, to_update=True, color='red', label=model_str)
g.draw_graph(coord_x, model(w_2, coord_x), 1, to_update=True, color='red', label=model_str)

plt.pause(1)
for i in range(n_iter):
    k = np.random.randint(0, sz - batch_size - 1)  # sz - размер выборки (массива coord_x)

    x_train = coord_x[k: k + batch_size]
    y_train = coord_y[k: k + batch_size]

    Qk_1 = np.average([loss(w_1, x_i, y_i) for x_i, y_i in zip(x_train, y_train)])
    Qe_1 = lm * Qk_1 + (1-lm) * Qe_1

    grad = np.mean([dL(w_1, x_i, y_i) for x_i, y_i in zip(x_train, y_train)], axis=0)
    w_1 = w_1 - eta * (grad + lm_l1*np.sign(w_1)*np.array([0, 1, 1, 1, 1]))

    new_y1 = model(w_1, coord_x)
    updata_formula = f"$a(x) = {w_1[0]:.2f} {w_1[1]:+.2f}x {w_1[2]:+.2f}x^2 {w_1[3]:+.2f}x^3 {w_1[4]:+.3f}x^4$"
    g.update_graphs(coord_x, new_y1, 0, new_label=updata_formula)
    g.update_text(0, f'Итерация: {i + 1} / {n_iter}')
    g.update_text(1, f"Q = {Qe_1:.3f}")

    # Без L1 :

    Qk_2 = np.average([loss(w_2, x_i, y_i) for x_i, y_i in zip(x_train, y_train)])
    Qe_2 = lm * Qk_2 + (1 - lm) * Qe_2

    grad = np.mean([dL(w_2, x_i, y_i) for x_i, y_i in zip(x_train, y_train)], axis=0)
    w_2 = w_2 - eta * grad

    new_y2 = model(w_2, coord_x)
    updata_formula = f"$a(x) = {w_2[0]:.2f} {w_2[1]:+.2f}x {w_2[2]:+.2f}x^2 {w_2[3]:+.2f}x^3 {w_2[4]:+.3f}x^4$"
    g.update_graphs(coord_x, new_y2, 1, new_label=updata_formula)
    g.update_text(2, f"Q = {Qe_2:.3f}")

    g.update_legend(0, 'lower right')
    g.update_legend(1, 'lower right')

    plt.pause(0.05)

Q_1 = np.average(loss(w_1, coord_x, coord_y))
Q_2 = np.average(loss(w_2, coord_x, coord_y))


print(f"w_1 = {w_1}")
print(f"Q_1 = {Q_1}", end='\n\n')

print(f"w_2 = {w_2}")
print(f"Q_2 = {Q_2}")


plt.show()