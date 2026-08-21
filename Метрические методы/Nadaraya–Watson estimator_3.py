import numpy as np

rub_usd = np.array([75, 76, 79, 82, 85, 81, 83, 86, 87, 85, 83, 80, 77, 79, 78, 81, 84])
n_data = len(rub_usd)
h = 3

# Создаём список, который будет пополняться
all_values = list(rub_usd)
days = np.arange(1, n_data + 1)  # исходные дни

predict = []

for day in range(n_data + 1, n_data + 11):
    # Расстояния до всех известных дней
    r = np.abs(day - np.arange(1, len(all_values) + 1)) / h
    K = (1 / np.sqrt(2 * np.pi)) * np.exp(-r**2 / 2)

    # Прогноз
    y_pred = np.sum(np.array(all_values) * K) / np.sum(K)

    # Сохраняем прогноз
    predict.append(y_pred)

    # Добавляем прогноз в выборку для следующих шагов
    all_values.append(y_pred)
