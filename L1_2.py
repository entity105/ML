import numpy as np
import matplotlib.pyplot as plt

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
w = np.zeros(N) # начальные нулевые значения параметров модели
n_iter = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20 # размер мини-батча (величина K = 20)

Qe = np.average(loss(w, coord_x, coord_y))   # начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

for _ in range(n_iter):
    k = np.random.randint(0, sz - batch_size - 1)  # sz - размер выборки (массива coord_x)

    x_train = coord_x[k: k + batch_size]
    y_train = coord_y[k: k + batch_size]

    Qk = np.average([loss(w, x_i, y_i) for x_i, y_i in zip(x_train, y_train)])
    Qe = lm * Qk + (1-lm) * Qe

    grad = np.mean([dL(w, x_i, y_i) for x_i, y_i in zip(x_train, y_train)], axis=0)
    w = w - eta * (grad + lm_l1*np.sign(w)*np.array([0, 1, 1, 1, 1]))

Q = np.average(loss(w, coord_x, coord_y))

w_str = [f"{val:+.2f}" for val in w]
formula = rf'$y = {w_str[0][1:]} {w_str[1]}x {w_str[2]}x^2 {w_str[3]}x^3 {w_str[4]}x^4$'

plt.plot(coord_x, coord_y)
plt.plot(coord_x, model(w, coord_x))
plt.legend([r'$y = -0.5x^2 + 0.1x^3 + \cos(3x) + 7$', formula])

print(w)
print(Q)

plt.show()