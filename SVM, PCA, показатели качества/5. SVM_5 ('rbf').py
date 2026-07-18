import numpy as np
from sklearn import svm


def func(x):
    return np.sin(0.5*x) + 0.2 * np.cos(2*x) - 0.1 * np.sin(4 * x) + 3


# обучающая выборка
coord_x = np.expand_dims(np.arange(-4.0, 6.0, 0.1), axis=1)
coord_y = func(coord_x).ravel()

svr = svm.SVR(kernel='rbf')    # Нелинейное ядро
x_train = coord_x[::3]
y_train = coord_y[::3]

svr.fit(x_train, y_train)
predict = svr.predict(coord_x)

Q = np.mean((predict - coord_y)**2)
print(Q)