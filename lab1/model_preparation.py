from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

# Загрузка предобработанных данных
train_data_scaled = pd.read_csv("train/train_data_scaled.csv")

# Подготовка данных для обучения
X_train = train_data_scaled[["X_scaled"]]
y_train = train_data_scaled["y"]

# Создание и обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Сохранение модели
joblib.dump(model, "model.pkl")
