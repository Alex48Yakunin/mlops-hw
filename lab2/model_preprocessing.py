from sklearn.preprocessing import StandardScaler
import pandas as pd


# Загрузка данных
train_data = pd.read_csv("train/train_data.csv")
test_data = pd.read_csv("test/test_data.csv")


# Предобработка данных
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data[["X", "y"]])
test_data_scaled = scaler.transform(test_data[["X", "y"]])

# Сохранение предобработанных данных
pd.DataFrame(train_data_scaled, columns=["X_scaled", "y"]).to_csv(
    "train/train_data_scaled.csv", index=False
)
pd.DataFrame(test_data_scaled, columns=["X_scaled", "y"]).to_csv(
    "test/test_data_scaled.csv", index=False
)
