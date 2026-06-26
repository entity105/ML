import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = [1, -2]
mean2 = [1, 3]
r = 0.7
D = 2.0
V = [[D, D * r], [D * r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])

# вычисление оценок МО и ковариационной матрицы
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = np.hstack([(x1.T - mm1).T, (x2.T - mm2).T])
VV = np.array([[np.dot(a[0], a[0]) / (2*N), np.dot(a[0], a[1]) / (2*N)],
                [np.dot(a[1], a[0]) / (2*N), np.dot(a[1], a[1]) / (2*N)]])

prior_y1 = prior_y2 = 0.5
l1 = l2 = 1

def a_y(x, m, prior, fine, cov_matrix=VV):
    inv_cov = np.linalg.inv(cov_matrix)
    alpha = inv_cov @ m
    betta = np.log(fine * prior) - 1/2 * (m.T @ inv_cov @ m)
    return alpha @ x.T + betta

def decision(x):
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Неверный формат данных")

    return [np.argmax([a_y(x_i, mm1, prior_y1, l1), a_y(x_i, mm2, prior_y2, l2)])*2-1 for x_i in x]

predict = decision(x_train)
Q = sum(int(a != y) for a, y in zip(predict, y_train))
print(Q)