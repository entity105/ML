import numpy as np
from graphs import ClassificationPlot
from matplotlib import pyplot as plt

np.random.seed(0)

# исходные параметры распределений трех классов
r1 = 0.7
D1 = 3.0
mean1 = [1, -2]
V1 = [[D1, D1 * r1], [D1 * r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [-3, -1]
V2 = [[D2, D2 * r2], [D2 * r2, D2]]

r3 = 0.3
D3 = 1.0
mean3 = [1, 2]
V3 = [[D3, D3 * r3], [D3 * r3, D3]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T
x3 = np.random.multivariate_normal(mean3, V3, N).T

x_train = np.hstack([x1, x2, x3]).T
y_train = np.hstack([np.zeros(N), np.ones(N), np.ones(N) * 2])

mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
mm3 = np.mean(x3.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x3.T - mm3).T
VV3 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# параметры для гауссовского байесовского классификатора
Py1, Py2, Py3 = 0.2, 0.5, 0.3
L1, L2, L3 = 1, 1, 1

def a_y(x, m, cov_matrix, prior, l):
    inv_cov = np.linalg.inv(cov_matrix)
    log_const = np.log(1/np.sqrt(2*np.pi*np.linalg.det(cov_matrix)))
    log_exp = -0.5 * ((x - m).T @ inv_cov @ (x - m))
    log_prior = np.log(l * prior)
    return log_const + log_exp + log_prior

def classificator(X):
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("Неверный формат данных")

    return [np.argmax([a_y(x_i, mm1, VV1, Py1, L1),
                      a_y(x_i, mm2, VV2, Py2, L2),
                      a_y(x_i, mm3, VV3, Py3, L3)]) for x_i in X]

predict = classificator(x_train)
Q = sum(int(a != y) for a, y in zip(predict, y_train))
print(Q)

c = ClassificationPlot("Наивная байесовская классификация", figsize=(10, 6))
settings = ({"label":"Класс 1 (y = 1)", "s":60},
            {"label":"Класс 2 (y = -1)", "s":60},
            {"label":"Класс 3", "s":100, "linewidths":2})
mistakes = [2 if x != y else x for x, y in zip(predict, y_train)]

c.draw_cls_points(x_train, y_train, settings)
# c.set_text_axis(f"$M[y=1] = ({mx11:.2f}, {mx12:.2f})$\n"
#                 f"$M[y=-1] = ({mx21:.2f}, {mx22:.2f})$\n\n"
#                 f"$D[y=1] = ({Dx11:.2f}, {Dx12:.2f})$\n"
#                 f"$D[y=-1] = ({Dx21:.2f}, {Dx22:.2f})$", coord_text=(0.02, 0.96))

c.set_text_axis(f"Кол-во ошибок: {mistakes.count(2)}\n"
                f"Q = {Q}", coord_text=(0.02, 0.7))


c.update_legend()
plt.show()