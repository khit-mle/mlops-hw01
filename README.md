# mlops-hw01
Repository for MLOps course 🤖 Homework No. 1

Репозиторий для ДЗ №1 по MLOps

Для выполнения задания была взята идея генерации синтетических данных по цене акций двух воображаемых компаний на фондовой бирже.

[data_creation.py](./data_creation.py) генерирует данные по следующим переменным:
- date,
- {company}_stock_price,
- {company}_stock_price_30d_moving_average,
- high_tech_comp_index_30d_moving_average,
- twitter_sentiment,
- web_search_interest_trend.

Предобработка данных осуществляется с помощью скрипта [model_preprocessing.py](./model_preprocessing.py)

В [model_preparation.py](./model_preparation.py) с помощью библиотек `TensorFlow` и `Keras` создаём модель сети долгой краткосрочной памяти (long short-term memory; LSTM) на основе данных из `train`.

[model_testing.py](./model_testing.py) проверяет модель машинного обучения на построенных данных из директории `test`.

[pipeline.sh](./pipeline.sh) последовательно запускает все представленные скрипты и в результате его успешного выполнения мы получим следующую локальную структуру проекта:

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
├── pyproject.toml
├── README.md
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
    └── xyz_train.npz

7 directories, 31 files
```
