import numpy as np


def func(x):
    return 0.5 * x + 0.2 * x ** 2 - 0.05 * x ** 3 + 0.2 * np.sin(4 * x) - 3


coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)
coord_y = coord_y - np.mean(coord_y)  # центрирование функции

K = 10
X = np.array([[xx**i for i in range(K)] for xx in coord_x]) # обучающая выборка для поиска коэффициентов модели
Y = coord_y

X_train = X[::2]  # обучающая выборка (входы)
Y_train = Y[::2]  # обучающая выборка (целевые значения)

F = (X_train.T @ X_train) / len(X)      # Матрица Грамма
L, W = np.linalg.eig(F)                 # СЗ и СВ
sort_ind = np.argsort(L)[::-1]
WW = W[:, sort_ind]                     # Новый базис (матрица собственных векторов)

G = (X @ WW)[:, :7]                     # Перешли в новое признаковое пространство и убрали 3 последних признака
XX_train = G[::2]                       # Обучающая выборка в новом пространстве

w = np.linalg.inv(XX_train.T @ XX_train) @ XX_train.T @ Y_train
predict = G @ w


from matplotlib import pyplot as plt
fig, ax = plt.subplots()
ax.grid()
ax.plot(coord_x, coord_y)
ax.plot(coord_x, predict)

plt.show()