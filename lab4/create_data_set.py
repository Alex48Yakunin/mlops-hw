from catboost.datasets import titanic
import pandas as pd

# Создание датасета
titanic_train, titanic_test = titanic()
titanic_df = pd.concat([titanic_train, titanic_test], ignore_index=True)
titanic_df.to_csv('datasets/data.csv', index=False)

# # Модификация датасета
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)

# # Сохранение изменений в файл
titanic_df.to_csv('./datasets/titanic_modified.csv', index=False)

# Заполнение пропущенных значений средним значением
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)

# Сохранение изменений в файл
titanic_df.to_csv('./datasets/titanic_filled_age.csv', index=False)
