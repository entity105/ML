import numpy as np

# исходная функция, которую нужно аппроксимировать моделью a(x)
def func(x):
    return -0.7 * x - 0.2 * x ** 2 + 0.05 * x ** 3 - 0.2 * np.cos(3 * x) + 2

def model(w, x):
    return w @ x.T

def Q(w, x, y):
    return np.mean((model(w, x) - y) ** 2)

def dQ_dw(w, x, y):
    return (2 / batch_size) * x.T @ (x @ w - y)


coord_x = np.arange(-4.0, 6.0, 0.1) # значения по оси абсцисс [-4; 6] с шагом 0.1
coord_y = func(coord_x) # значения функции по оси ординат

sign_X = np.column_stack([np.ones_like(coord_x), coord_x, coord_x**2, coord_x**3])  # n x 4

sz = len(coord_x)	# количество значений функций (точек)
eta = np.array([0.1, 0.01, 0.001, 0.0001]) # шаг обучения для каждого параметра w0, w1, w2, w3
w = np.array([0., 0., 0., 0.]) # начальные значения параметров модели
N = 500 # число итераций алгоритма SGD
lm = 0.02 # значение параметра лямбда для вычисления скользящего экспоненциального среднего
batch_size = 20 # размер мини-батча (величина K = 20)
gamma = 0.8 # коэффициент гамма для вычисления импульсов Нестерова
v = np.zeros(len(w))  # начальное значение [0, 0, 0, 0]

Qe = Q(w, sign_X, coord_y)# начальное значение среднего эмпирического риска
np.random.seed(0) # генерация одинаковых последовательностей псевдослучайных чисел

for i in range(N):
    k = np.random.randint(0, sz - batch_size)

    selected_x = sign_X[k : k + batch_size]   # K x 4
    selected_y = coord_y[k : k + batch_size]

    v = gamma * v + (1 - gamma)*eta*dQ_dw(w - gamma * v, selected_x, selected_y)
    w = w - v
    Qe = lm*Q(w, selected_x, selected_y) + (1-lm)*Qe

Q = Q(w, sign_X, coord_y)
print(w)