import numpy as np
from matplotlib import pyplot as plt
from graphs import DynamicGraphs


# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.1 * x**2 - np.sin(x) + 5.

def model(w, x):
    """Если x - число, вернёт число (предсказание для 1 точки); если x - массив, то вернёт массив предсказаний для набора x"""
    return w[0] + w[1]*x + w[2]*x**2 + w[3]*x**3

# Эту функцию будем минимизировать
def Q(a_x, y):
    """a_x - набор предсказаний модели; y - целевые значения"""
    return np.average((a_x - y)**2)

# Вычисляем градиент по набору параметров w (w - массив)
def dQ(w, x, y) -> np.array:
    """w - массив; y - целевые значения, x - набор признаков"""
    S = np.column_stack([np.ones(sz), x, x ** 2, x ** 3])   # n x 4
    return (2/sz) * (w @ S.T - y) @ S   # -> (w_1, w_2, w_3, w_4)


coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 200 # число итераций градиентного алгоритма
Q_0 = Q(model(w, coord_x), coord_y)

# Визуализация

g = DynamicGraphs("Аппроксимация функции", figsize=(10, 6))
g.set_default_text_fig(N, '$model: a(x) = w_0 + w_1x + w_2x^2 + w_3x^3$')   # 0
g.set_text_axis(f"Q = {Q_0:.3f}", coord_text=(0.03, 0.95), to_updata=True)     # 1

formula_1 = r'$y = 0.1x^2 - sin(x) + 5$'
g.drow_graph(coord_x, coord_y, base_setting=False, color='blue', linewidth=3, label=formula_1)

formula_2 = f"{w[0]} + {w[1]}x + {w[2]}x^2 + {w[3]}x^3"
g.drow_graph(coord_x, model(w, coord_x) + 4, to_updata=True, base_setting=False, color='red', linewidth=3, label=formula_2)

plt.pause(1)
for i in range(N):
    w = w - eta * dQ(w, coord_x, coord_y)
    y = model(w, coord_x)
    Q_i = Q(y, coord_y)

    update_formula = f"$a(x) = {w[0]:.2f} + {w[1]:.2f}x + {w[2]:.2f}x^2 + {w[3]:.2f}x^3$"
    g.updata_graphs(coord_x, y, new_label=update_formula)
    g.update_text(0, f'Итерация: {i+1} / {N}')
    g.update_text(1, f"Q = {Q_i:.3f}")
    plt.pause(0.1)

Q = Q(model(w, coord_x), coord_y)

print(w)
print(Q)

plt.show()