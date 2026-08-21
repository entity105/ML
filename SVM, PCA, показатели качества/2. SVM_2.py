import matplotlib.pyplot as plt
import numpy as np
from graphs import DynamicGraphs
from sklearn import svm

def func(x):
    return np.sin(0.5*x) + 0.2 * np.cos(2*x) - 0.1 * np.sin(4 * x) - 2.5


def model(w, x):
    return w[0] + w[1] * x + w[2] * x ** 2 + w[3] * x ** 3 + w[4] * np.cos(x) + w[5] * np.sin(x)


# обучающая выборка
coord_x = np.arange(-4.0, 6.0, 0.1)
coord_y = func(coord_x)

x_train = np.array([[x, x**2, x**3, np.cos(x), np.sin(x)] for x in coord_x])
y_train = coord_y

svr = svm.SVR(kernel='linear')
svr.fit(x_train, y_train)

w = [svr.intercept_[0], *svr.coef_[0]]
predict = svr.predict(x_train)

Q = np.mean((y_train - predict)**2)

# print(w)
# print(predict)

g = DynamicGraphs("SVR", figsize=(10, 6))
g.draw_graph(coord_x, coord_y, label="f(x) = sin(0.5x) + 0.2cos(2x) - 0.1sin(4x) - 2.5")
g.draw_graph(coord_x, predict,
             label=f"$g(x) = {w[0]:+.2f}x {w[1]:+.2f}x^2 {w[2]:+.2f}x^3 {w[3]:+.2f}cos(x) {w[4]:+.2f}sin(x)$",
             color="red")
g.set_text_axis(text=f"Q = {Q:.3f}")
plt.show()