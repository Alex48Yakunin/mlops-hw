from sklearn.metrics import mean_squared_error
import pandas as pd
import joblib

# Загрузка модели
model = joblib.load("model.pkl")

# Загрузка тестовых данных
test_data_scaled = pd.read_csv("test/test_data_scaled.csv")
X_test = test_data_scaled[["X_scaled"]]
y_test = test_data_scaled["y"]

# Предсказание и оценка модели
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"MSE: {mse}")
