import numpy as np
from graphs import ClassificationPlot
from matplotlib import pyplot as plt

np.random.seed(0)

# исходные параметры распределений двух классов
r1 = 0.7
D1 = 1.0
mean1 = [-1, -2, -1]
V1 = [[D1, D1 * r1, D1*r1*r1], [D1 * r1, D1, D1*r1], [D1*r1*r1, D1*r1, D1]]

r2 = 0.5
D2 = 2.0
mean2 = [1, 2, 1]
V2 = [[D2, D2 * r2, D2*r2*r2], [D2 * r2, D2, D2*r2], [D2*r2*r2, D2*r2, D2]]

# моделирование обучающей выборки
N = 1000
x1 = np.random.multivariate_normal(mean1, V1, N).T
x2 = np.random.multivariate_normal(mean2, V2, N).T
# print(x1.shape)

x_train = np.hstack([x1, x2]).T
y_train = np.hstack([np.ones(N) * -1, np.ones(N)])
# print(x_train.shape)

# матожидания
M_y1 = np.mean(x1.T, axis=0)
M_y2 = np.mean(x2.T, axis=0)
# print(M_y1.shape)

# ковариационные матрицы
VV1 = np.cov(x1, ddof=0)
VV2 = np.cov(x2, ddof=0)

prior_y1 = prior_y2 = 0.5
l1 = l2 = 1

def a_y(x, cov_matrix, m, prior):
    det_cov = np.linalg.det(cov_matrix)
    inv_cov = np.linalg.inv(cov_matrix)
    log_const = np.log(1/np.sqrt(det_cov))
    log_exp = 1/2 * ((x - m).T @ inv_cov @ (x - m))
    log_py1 = np.log(prior)
    return log_const - log_exp + log_py1

def decision(data: np.array):
    if data.ndim != 2 or data.shape[1] != 3: # (3 признака)
        raise ValueError("Некорректные данные")

    return np.array([np.argmax([a_y(x_i, VV1, M_y1, prior_y1), a_y(x_i, VV2, M_y2, prior_y2)])*2-1 for x_i in data])

predict = list(decision(x_train))
Q = sum(int(a != y) for a, y in zip(predict, y_train))
# print(Q)

c = ClassificationPlot("Наивная байесовская классификация", figsize=(10, 6))
settings = ({"label":"Класс 1 (y = 1)", "s":60},
            {"label":"Класс 2 (y = -1)", "s":60},
            {"label":"Ошибка", "marker":"x", "s":100, "linewidths":2})
mistakes = [2 if x != y else x for x, y in zip(predict, y_train)]

c.draw_cls_points(x_train, mistakes, settings)
# c.set_text_axis(f"$M[y=1] = ({mx11:.2f}, {mx12:.2f})$\n"
#                 f"$M[y=-1] = ({mx21:.2f}, {mx22:.2f})$\n\n"
#                 f"$D[y=1] = ({Dx11:.2f}, {Dx12:.2f})$\n"
#                 f"$D[y=-1] = ({Dx21:.2f}, {Dx22:.2f})$", coord_text=(0.02, 0.96))

c.set_text_axis(f"Кол-во ошибок: {mistakes.count(2)}\n"
                f"Q = {Q}", coord_text=(0.02, 0.7))


c.update_legend()
plt.show()