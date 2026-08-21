import numpy as np
from sklearn import svm

def func(x):
    return np.sin(0.5*x) + 0.9 * np.cos(2*x) - 0.1 * np.sin(4 * x) - 2.5


def model(w, x):
    return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3 + w[4] * np.cos(x) + w[5] * np.sin(x)


# обучающая выборка
coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

x_train = np.array([[x, x**2, x**3, np.cos(x), np.sin(x)] for x in coord_x])
y_train = coord_y

svr = svm.SVR(kernel='linear')
svr.fit(x_train, y_train)
w1 = svr.coef_[0]
w0 = svr.intercept_[0]
w = np.concatenate([[w0], w1])

Q = np.mean((model(w, coord_x) - coord_y)**2)
print(Q)
Q1 = np.mean((svr.predict(x_train) - coord_y)**2)
print(Q1)