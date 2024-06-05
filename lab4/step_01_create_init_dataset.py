import os
from catboost.datasets import titanic


# Проверка и создание директории, если она не существует
data_dir = 'lab4/data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Загрузка датасета
train_df, _ = titanic()

# Сохранение нужных столбцов
initial_df = train_df[['Pclass', 'Sex', 'Age']]

# Сохранение датасета в CSV файл
initial_df.to_csv(os.path.join(data_dir, 'titanic_initial.csv'), index=False, header=True)
