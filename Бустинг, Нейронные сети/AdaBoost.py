import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


np.random.seed(0)
n_feature = 2

# исходные параметры для формирования образов обучающей выборки
r1 = 0.7
D1 = 3.0
mean1 = [3, 7]
V1 = [[D1 * r1 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

r2 = 0.5
D2 = 2.0
mean2 = [4, 2]
V2 = [[D2 * r2 ** abs(i-j) for j in range(n_feature)] for i in range(n_feature)]

# моделирование обучающей выборки
N1, N2 = 1000, 1200
x1 = np.random.multivariate_normal(mean1, V1, N1).T
x2 = np.random.multivariate_normal(mean2, V2, N2).T

data_x = np.hstack([x1, x2]).T
data_y = np.hstack([np.ones(N1) * -1, np.ones(N2)])

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, random_state=123,test_size=0.3, shuffle=True)

T = 10
max_depth = 3
w = np.ones(len(x_train)) / len(x_train) # начальные значения весов для объектов выборки
algorithms = []
alphas = []

for t in range(T):
    b_t = DecisionTreeClassifier(criterion='gini', max_depth=max_depth)
    b_t.fit(x_train, y_train, sample_weight=w)

    b_t_x = b_t.predict(x_train)
    # N_bt = np.sum(w * b_t_x != y_train)
    N = np.sum(np.abs(y_train - b_t_x) / 2 * w)

    a_t = 0.5 * np.log((1 - N) / N) if not (-0.0001 <= N <= 0.0001) else np.log((1-1e-8) / 1e-8)

    w = w * np.exp(-a_t * y_train * b_t_x)
    w = w / np.sum(w)

    algorithms.append(b_t)
    alphas.append(a_t)
alphas = np.array(alphas)

def a_x(X):
    predicts = np.array([algorithm.predict(X) for algorithm in algorithms])
    ab = np.array([alpha * pred for alpha, pred in zip(alphas, predicts)])
    summ = np.sum(ab, axis=0)
    return np.sign(summ)

predict = list(a_x(x_test))
Q = np.sum(predict != y_test)
print(Q)