import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.7
D1 = 1.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [1, 3]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

# обучающая выборка для байесовского классификатора (стандартный формат)
x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])

# вычисление оценок математических ожиданий
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

# вычисление ковариационных матриц
a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# параметры для гауссовского байесовского классификатора
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации

def a_y(x, cov_matrix, m, prior):
    det_cov = np.linalg.det(cov_matrix)
    inv_cov = np.linalg.inv(cov_matrix)
    log_const = np.log(1/np.sqrt(det_cov))
    log_exp = 1/2 * ((x - m).T @ inv_cov @ (x - m))
    log_py1 = np.log(prior)
    return log_const - log_exp + log_py1

def decision(data: np.array):
    if data.ndim != 2 or data.shape[1] != 2: # (2 признака)
        raise ValueError("Некорректные данные")

    return np.array([np.argmax([a_y(x_i, VV1, mm1, Py1), a_y(x_i, VV2, mm2, Py2)])*2-1 for x_i in data])

predict = list(decision(x_train))
Q = sum(int(a != y) for a, y in zip(predict, y_train))

print(Q)

# График
# print(x_train.T[0].shape, y_train.shape)
# plt.scatter(x_train.T[0], y_train)
# plt.show()