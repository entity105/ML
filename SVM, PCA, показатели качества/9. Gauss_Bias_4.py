import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.7
D1 = 1.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = -0.5
D2 = 2.0
mean2 = [0, 2]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N1 = 500
N2 = 1000
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

# вычисление оценок МО и ковариационных матриц
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N1, np.dot(a[0], a[1]) / N1],
                [np.dot(a[1], a[0]) / N1, np.dot(a[1], a[1]) / N1]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N2, np.dot(a[0], a[1]) / N2],
                [np.dot(a[1], a[0]) / N2, np.dot(a[1], a[1]) / N2]])

# для гауссовского байесовского классификатора
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации

def p_i(x, p_y, μ_y, cov_y, λ_y):
    cov_mtrx_inv = np.linalg.inv(cov_y)
    cov_mtrx_det = np.linalg.det(cov_y)
    term_1 = np.log(λ_y*p_y)
    term_2 = -0.5*(x-μ_y).T @ cov_mtrx_inv @ (x-μ_y)
    term_3 = -0.5*np.log(cov_mtrx_det)
    return term_1 + term_2 + term_3

def model(p: np.ndarray|list) -> int:
    return np.argmax(p)*2-1

predict = np.array([model([p_i(x, Py1, mm1, VV1, L1), p_i(x, Py2, mm2, VV2, L2)]) for x in data_x])

TP = np.sum((predict == 1) & (data_y == 1))
TN = np.sum((predict == -1) & (data_y == -1))
FP = np.sum((predict == 1) & (data_y == -1))
FN = np.sum((predict == -1) & (data_y == 1))

print(TP, TN, FP, FN)
print(TP + TN + FP + FN)