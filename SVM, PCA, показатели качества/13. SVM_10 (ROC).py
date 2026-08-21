import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

np.random.seed(0)

# исходные параметры распределений классов
r1 = -0.2
D1 = 3.0
mean1 = [1, -5]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-1, -2]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

# моделирование обучающей выборки
N1 = 1000
N2 = 1000
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.5, shuffle=True)

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

w = [clf.intercept_[0], *clf.coef_[0]]
FPR = []
TPR = []

range_t = np.arange(5.7, -7.8, -0.1)
predict = clf.decision_function(x_test)    # Отступы
# pr = clf.predict(x_test)

for t in range_t:
    pred_t = np.where(predict >= t, 1, -1)

    FP = np.sum((pred_t == 1) & (y_test == -1))
    TN = np.sum((pred_t == -1) & (y_test == -1))
    TP = np.sum((pred_t == 1) & (y_test == 1))
    FN = np.sum((pred_t == -1) & (y_test == 1))

    FPR_i = FP / (FP + TN)
    TPR_i = TP / (TP + FN)

    FPR.append(FPR_i)
    TPR.append(TPR_i)

plt.plot(FPR, TPR)


plt.show()