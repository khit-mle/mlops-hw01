# Репозиторий для ДЗ (лабораторных) по MLOps 🤖

## Содержание:
- [ДЗ №1](#дз-№1)

-------------------------
### [ДЗ №1](./lab1)

Для выполнения задания была взята идея генерации синтетических данных по цене акций двух воображаемых компаний на фондовой бирже.

[data_creation.py](./lab1/data_creation.py) генерирует данные по следующим переменным:
- date,
- {company}_stock_price,
- {company}_stock_price_30d_moving_average,
- high_tech_comp_index_30d_moving_average,
- twitter_sentiment,
- web_search_interest_trend.

Предобработка данных осуществляется с помощью скрипта [model_preprocessing.py](./lab1/model_preprocessing.py)

В [model_preparation.py](./lab1/model_preparation.py) с помощью библиотек `TensorFlow` и `Keras` создаём модель сети долгой краткосрочной памяти (long short-term memory; LSTM) на основе данных из `train`.

[model_testing.py](./lab1/model_testing.py) проверяет модель машинного обучения на построенных данных из директории `test`.

[pipeline.sh](./lab1/pipeline.sh) последовательно запускает все представленные скрипты и в результате его успешного выполнения мы получим следующую локальную структуру в `lab1/`:

```bash
.
├── data_creation.py
├── model_preparation.py
├── model_preprocessing.py
├── models
│   ├── abc_evaluation.png
│   ├── abc_lstm_model.h5
│   ├── xyz_evaluation.png
│   └── xyz_lstm_model.h5
├── model_testing.py
├── pipeline.sh
├── plots
│   ├── ABC
│   │   ├── ABC_stock_price_30d_moving_average.png
│   │   ├── ABC_stock_price.png
│   │   ├── high_tech_comp_index_30d_moving_average.png
│   │   ├── twitter_sentiment.png
│   │   └── web_search_interest_trend.png
│   └── XYZ
│       ├── high_tech_comp_index_30d_moving_average.png
│       ├── twitter_sentiment.png
│       ├── web_search_interest_trend.png
│       ├── XYZ_stock_price_30d_moving_average.png
│       └── XYZ_stock_price.png
├── requirements.txt
├── test
│   ├── abc_test.csv
│   ├── abc_test.npz
│   ├── xyz_test.csv
│   └── xyz_test.npz
└── train
    ├── abc_scaler.pkl
    ├── abc_train.csv
    ├── abc_train.npz
    ├── xyz_scaler.pkl
    ├── xyz_train.csv
    └── xyz_train.np
```
