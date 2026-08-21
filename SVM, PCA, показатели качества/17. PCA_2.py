import numpy as np

np.random.seed(0)

n_total = 1000 # число образов выборки
n_features = 200 # число признаков

table = np.zeros(shape=(n_total, n_features))
lmd = 0.01

for _ in range(100):
    i, j = np.random.randint(0, n_total), np.random.randint(0, n_features)
    table[i, j] = np.random.randint(1, 10)

F = 1/n_total * table.T @ table
L, W = np.linalg.eig(F)

ind_sort = np.argsort(L)[::-1]
LL = L[ind_sort]
WW = W[:, ind_sort]

data_x = table @ WW
data_x = data_x[:, LL >= lmd]
print(data_x)
print(data_x.shape)