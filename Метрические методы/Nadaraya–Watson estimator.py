import numpy as np

# координаты четырех точек
x = np.array([0, 1, 2, 3])
y = np.array([0.5, 0.8, 0.6, 0.2])

x_est = np.arange(0, 3.1, 0.1) # множество точек для промежуточного восстановления функции
h = 1

y_est = []
for x_i in x_est:
    distances = np.abs(x_i - x)
    K = np.where(distances <= 1, np.abs(1 - distances) / h, 0)

    res = np.sum(y * K) / np.sum(K)
    y_est.append(res)

print(y_est)