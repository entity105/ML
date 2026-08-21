import matplotlib.pyplot as plt
import numpy as np
from graphs import ClassificationPlot
from sklearn import svm
from sklearn.model_selection import train_test_split


np.random.seed(0)
# исходные параметры распределений классов
r1 = 0.6
D1 = 3.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.7
D2 = 2.0
mean2 = [-3, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

r3 = 0.5
D3 = 1.0
mean3 = [1, 2]
V3 = [[D3, D3 * r3], [D3 * r3, D3]]

# моделирование обучающей выборки
N = 500
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T
x3 = np.random.multivariate_normal(mean3, V3, N).T

data_x = np.hstack([x1, x2, x3]).T
data_y = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.3, shuffle=True)
# print(x_train)
# print(y_train)

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

w1 = clf.coef_[0]
w2 = clf.coef_[1]
w3 = clf.coef_[2]

w01 = clf.intercept_[0]
w02 = clf.intercept_[1]
w03 = clf.intercept_[2]

w1 = np.concatenate([[w01], w1])
w2 = np.concatenate([[w02], w2])
w3 = np.concatenate([[w03], w3])

predict = clf.predict(x_test)
Q = np.sum(predict != y_test)
print(Q)

c_plt = ClassificationPlot("SVC")
c_plt.draw_cls_points(data_x, (0, 1, 2))

plt.show()