import numpy as np

x = np.arange(-2, 3, 0.1)
y = -x + 0.2 * x ** 2 - 0.5 * np.sin(4*x) + np.cos(2*x)

t = 0

left_mask = x < t
right_mask = ~left_mask

R_1 = left_mask.sum()
b_1 = np.mean(y[left_mask])
HR1 = np.sum((b_1 - y[left_mask])**2)

R_2 = right_mask.sum()
b_2 = np.mean(y[right_mask])
HR2 = np.sum((b_2 - y[right_mask])**2)

R = len(y)
b = np.mean(y)
HR = np.sum((b - y)**2)

IG = HR - R_1 / R * HR1 - R_2 / R * HR2