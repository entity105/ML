import numpy as np


# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return -0.7 * x - 0.2 * x ** 2 + 0.05 * x ** 3 - 0.2 * np.cos(3 * x) + 2


# здесь объявляйте необходимые функции
def model(w, x):
    return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3


def loss(w, x, y):
    return (model(w, x) - y) ** 2


def dL(w, x, y):
    return 2 * (model(w, x) - y) * np.array([1, x, x ** 2, x ** 3])


coord_x = np.arange(-4.0, 6.0, 0.1)  # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x)  # значения функции по оси ординат

sz = len(coord_x)  # количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001])  # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.])  # начальные значения параметров модели
N = 500  # число итераций алгоритма SGD
lm = 0.02  # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20  # размер мини-батча (величина K = 20)
gamma = 0.8  # коэффициент гамма для вычисления моментов Нестерова
v = np.zeros(len(w))  # начальное значение [0, 0, 0, 0]

Qe = np.mean(loss(w, coord_x, coord_y))
np.random.seed(0)  # генерация одинаковых последовательностей псевдослучайных чисел

# здесь продолжайте программу
for _ in range(N):
    k = np.random.randint(0, sz - batch_size - 1)

    Qg = 0
    grad = 0
    for i in range(k, k + batch_size):
        grad = grad + dL(w - gamma * v, coord_x[i], coord_y[i])
        Qg = Qg + loss(w, coord_x[i], coord_y[i])

    Qg = Qg / batch_size
    v = gamma * v + (1 - gamma) * eta * grad / batch_size
    w = w - v  # eta * grad / batch_size

    Qe = lm * Qg + (1 - lm) * Qe

Q = np.mean(loss(w, coord_x, coord_y))

print(w)