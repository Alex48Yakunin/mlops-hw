import numpy as np
import pandas as pd


# Функция генерации данных
def generate_data(n_samples, noise_std=0.1, anomaly_freq=0.01):
    np.random.seed(42)
    X = np.linspace(-30, 30, n_samples)
    y = np.sin(X) + noise_std * np.random.randn(n_samples)
    # Добавляем аномалии
    anomalies = np.random.choice(
        n_samples, size=int(n_samples * anomaly_freq), replace=False
    )
    y[anomalies] += 5  # Увеличение значений в аномальных точках
    return X, y


# Генерация данных
train_X, train_y = generate_data(1000)
test_X, test_y = generate_data(500)


# Сохранение данных
pd.DataFrame({"X": train_X, "y": train_y}).to_csv(
    "train/train_data.csv", index=False
    )
pd.DataFrame({"X": test_X, "y": test_y}).to_csv(
    "test/test_data.csv", index=False
    )
