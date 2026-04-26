import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x**2 - 0.1 * 1/np.exp(-x) + 0.5 * np.cos(2*x) - 2.

def model(w, x):
    return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * np.cos(2*x) + w[4] * np.sin(2*x)

def L(w, x_i, y_i):
    """Вычисляет ошибку для элемента или для массива"""
    return (model(w, x_i) - y_i)**2

def dLdw(w, x_i, y_i):
    """Градиент ошибки, принимает только числа для x_i y_i"""
    return 2 * (model(w, x_i) - y_i) * np.array([1, x_i, x_i**2, np.cos(2*x_i), np.sin(2*x_i)])

coord_x = np.arange(-5.0, 5.0, 0.1) # значения по оси абсцисс [-5; 5] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат


sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.01, 0.001, 0.0001, 0.01, 0.01]) # шаг обучения для каждого параметра w0, w1, w2, w3, w4
w = np.array([0., 0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего

Qe = np.average(L(w, coord_x, coord_y)) # начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

for i in range(N):
    k = np.random.randint(0, sz)  # sz - размер выборки (массива coord_x)
    x_i = coord_x[k]
    y_i = coord_y[k]
    w = w - eta * dLdw(w, x_i, y_i)
    epsilon_i = L(w, x_i, y_i)
    Qe = lm * epsilon_i + (1 - lm)*Qe

Q = np.average(L(w, coord_x, coord_y))