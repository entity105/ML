import numpy as np

np.random.seed(0)

# исходные параметры распределений двух классов
mean1 = np.array([1, -2, 0])
mean2 = np.array([1, 3, 1])
r = 0.7
D = 2.0
V = [[D, D * r, D*r*r], [D*r, D, D*r], [D*r*r, D*r, D]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V, N).T
x2 = np.random.multivariate_normal(mean2, V, N).T
# print(x1.shape)


x_train = np.hstack([x1, x2]).T     # 2000 x 3
y_train = np.hstack([np.zeros(N), np.ones(N)])
# print(x_train.shape)
# print(y_train)

# здесь вычисляйте векторы математических ожиданий и ковариационную матрицу по выборке x1, x2
mm1 = np.mean(x1, axis=1)
mm2 = np.mean(x2, axis=1)
# print(mm1)

VV = np.cov(x_train.T, ddof=0)
VV = (np.cov(x1) + np.cov(x2))/2
# print(VV.shape)
# print(VV)

# параметры для линейного дискриминанта Фишера
Py1, L1 = 0.5, 1  # вероятности появления классов
Py2, L2 = 1 - Py1, 1  # и величины штрафов неверной классификации

inv_VV = np.linalg.inv(VV)
alpha1 = inv_VV @ mm1
beta1 = np.log(L1 * Py1) - 0.5*(mm1.T @ inv_VV @ mm1)

alpha2 = inv_VV @ mm2
beta2 = np.log(L2 * Py2) - 0.5*(mm2.T @ inv_VV @ mm2)

print(alpha1)