import numpy as np
from sklearn.model_selection import train_test_split
from graphs import ClassificationPlot


def loss(w, x, y):
    return np.exp(-np.dot(w, x) * y)


def df(w, x, y):
    return -np.exp(-np.dot(w, x) * y) * x * y


np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.4
D1 = 2.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 3.0
mean2 = [2, 3]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T

data_x = np.array([[1, x[0], x[1]] for x in np.hstack([x1, x2]).T])
data_y = np.hstack([np.ones(N) * -1, np.ones(N)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.3, shuffle=True)

n_train = len(x_train)  # размер обучающей выборки
w = [0.0, 0.0, 0.0]  # начальные весовые коэффициенты
nt = np.array([0.5, 0.01, 0.01])  # шаг обучения для каждого параметра w0, w1, w2
N = 500  # число итераций алгоритма SGD
batch_size = 10 # размер мини-батча (величина K = 10)

# c = ClassificationPlot("SDG")

for i in range(N):
    k = np.random.randint(0, n_train - batch_size - 1)  # n_train - размер выборки (массива x_train)

    x = x_train[k:k+batch_size]
    y = y_train[k:k+batch_size]

    grad = np.mean([df(w, x_i, y_i) for x_i, y_i in zip(x, y)], axis=0)
    w = w - nt * grad

predict = w @ x_test.T

mrgs = predict * y_test
mrgs.sort()

acc = np.mean(np.sign(predict) == y_test)
