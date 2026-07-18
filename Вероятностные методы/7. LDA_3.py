import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = np.array([1, -2])
mean2 = np.array([-3, -1])
mean3 = np.array([1, 2])

r = 0.5
D = 1.0
V = [[D, D * r], [D*r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T
x3 = np.random.multivariate_normal(mean3, V, N).T

x_train = np.hstack([x1, x2, x3]).T
y_train = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

# здесь вычисляйте векторы математических ожиданий и ковариационную матрицу по выборке x1, x2, x3
mm1 = np.mean(x1, axis=1)
mm2 = np.mean(x2, axis=1)
mm3 = np.mean(x3, axis=1)

cov_matrix = (np.cov(x1) + np.cov(x2) + np.cov(x3)) / 3

# параметры для линейного дискриминанта Фишера
Py1, Py2, Py3 = 0.2, 0.4, 0.4
L1, L2, L3 = 1, 1, 1

def a_y(X, m, prior, l, cov=cov_matrix):
    inv_cov = np.linalg.inv(cov)
    α = inv_cov @ m
    β = np.log(l * prior) - 0.5*(m.T @ inv_cov @ m)
    return α @ X.T + β

def decision(X):
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("Неверный формат данных")

    return [np.argmax(
                      [a_y(x_i, mm1, Py1, L1),
                      a_y(x_i, mm2, Py2, L2),
                      a_y(x_i, mm3, Py3, L3)]
            ) for x_i in X]

predict = decision(x_train)
Q = sum(int(a != y) for a, y in zip(predict, y_train))
print(Q)