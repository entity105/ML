import numpy as np
from sklearn.cluster import DBSCAN

X = [(166, 88), (147, 119), (133, 147), (113, 175), (91, 92), (120, 126), (146, 151), (172, 174), (94, 192), (187, 193), (328, 82), (299, 97), (277, 131), (280, 171), (299, 198), (348, 194), (378, 153), (372, 95), (222, 169), (332, 141), (69, 256), (110, 258), (139, 257), (179, 256), (210, 256), (248, 256), (295, 256), (322, 254), (350, 252), (377, 251), (400, 247), (403, 260), (378, 278), (341, 273), (306, 274), (277, 275), (245, 274), (222, 275), (193, 276), (170, 276), (147, 279), (120, 274), (91, 275), (65, 279)]
X = np.array(X)

# clustering = DBSCAN(eps=55, min_samples=3, metric='euclidean')
# res = clustering.fit_predict(X)
#
# X1 = X[res == 0]
# X2 = X[res == 1]
# X3 = X[res == 2]
#
# Noise = X[res == -1]
#
# print(X1, X2, X3, Noise, sep='\n')

eps = 55
m = 3
U = X
N = set()
a = 0
while len(U) != 0:
    k = np.random.randint(0, len(U))
    x = U[k]

    distances = np.sqrt(np.sum((U - x)**2, axis=1))
    eps_neigboor = U[(distances < eps) & (distances != 0)]
    if len(eps_neigboor) < m:
        N.add(tuple(x))

    pass