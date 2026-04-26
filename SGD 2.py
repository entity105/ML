import numpy as np

# x_i, y_i - элемент (число)
# x, y - массивы (одномерные)
# X, Y - матрицы

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 2.5

def model(w, x):
    X = np.array([[1, x, x**2, x**3] for x in x])       # n x 4
    return w @ X.T                                      # 1 x n

def loss_func(w, x, y):
    return (model(w, x) - y)**2                         # 1 x n

def Q(w, x, y):
    return np.average(loss_func(w, x, y))               # 1 x 1

def dQ_dw(w, x, y, k):
    X = np.array([[1, x, x ** 2, x ** 3] for x in x])  # n x 4
    return 2/k * (model(w, x) - y) @ X                 # 1 x 4      ((1 x n - 1 x n) @ n x 4) = 1 x 4

coord_x = np.arange(-4.0, 6.0, 0.1) # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 50 # размер мини-батча (величина K = 50)

Qe = Q(w, coord_x, coord_y) # начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

for _ in range(N):
    k = np.random.randint(0, sz - batch_size)       # 0_[=====================|---------]_sz

    X_sect = coord_x[k: k + batch_size]                 # x_0[--------|============|---------]
    Y_sect = coord_y[k: k + batch_size]                 # y_0[--------|============|---------]

    w = w - eta * dQ_dw(w, X_sect, Y_sect, batch_size)
    Qe = lm*Q(w, X_sect, Y_sect) + (1 - lm) * Qe

Q = Q(w, coord_x, coord_y)
