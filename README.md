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

<details>
  <summary>👈 Вывод команды tree:</summary>

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
</details>

-----------------

<details>
  <summary>👈 Инструкция по запуску (проверялась на Debian 12):</summary>

```bash
mkdir test-lab-1

cd test-lab-1/

git clone https://github.com/khit-mle/mlops-practice.git .

cd lab1/

python3 -m virtualenv .venv

source .venv/bin/activate

pip3 install -r requirements.txt

bash pipeline.sh
```
</details>

-----------------

<details>
  <summary>👈 Пример вывода при запуске bash pipeline.sh</summary>

```bash
Running data_creation.py...

data_creation.py completed successfully.
Running model_preprocessing.py...

model_preprocessing.py completed successfully.
Running model_preparation.py...
2024-04-14 10:20:40.847702: I external/local_tsl/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-04-14 10:20:40.851345: I external/local_tsl/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-04-14 10:20:40.907878: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in pe
rformance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-04-14 10:20:42.044412: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
/home/debuser/test-lab-1/lab1/.venv/lib/python3.11/site-packages/keras/src/layers/rnn/rnn.py:204: UserWarning: Do not pass an `input_shape`/`input_dim` argu
ment to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
Epoch 1/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - loss: 0.0691 - root_mean_squared_error: 0.2541
Epoch 2/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0100 - root_mean_squared_error: 0.0997
Epoch 3/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0081 - root_mean_squared_error: 0.0900
Epoch 4/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0070 - root_mean_squared_error: 0.0838
Epoch 5/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0072 - root_mean_squared_error: 0.0847
Epoch 6/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0063 - root_mean_squared_error: 0.0790
Epoch 7/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0060 - root_mean_squared_error: 0.0777
Epoch 8/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0057 - root_mean_squared_error: 0.0753
Epoch 9/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0058 - root_mean_squared_error: 0.0758
Epoch 10/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0058 - root_mean_squared_error: 0.0760
Epoch 11/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0058 - root_mean_squared_error: 0.0762
Epoch 12/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0050 - root_mean_squared_error: 0.0708
Epoch 13/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0057 - root_mean_squared_error: 0.0756
Epoch 14/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0052 - root_mean_squared_error: 0.0719
Epoch 15/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0054 - root_mean_squared_error: 0.0731
Epoch 16/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0054 - root_mean_squared_error: 0.0732
Epoch 17/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0048 - root_mean_squared_error: 0.0693
Epoch 18/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0048 - root_mean_squared_error: 0.0690
Epoch 19/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0052 - root_mean_squared_error: 0.0720
Epoch 20/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0048 - root_mean_squared_error: 0.0691
Epoch 21/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0050 - root_mean_squared_error: 0.0704
Epoch 22/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0060 - root_mean_squared_error: 0.0773
Epoch 23/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0053 - root_mean_squared_error: 0.0724
Epoch 24/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0058 - root_mean_squared_error: 0.0759
Epoch 25/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0047 - root_mean_squared_error: 0.0685
Epoch 26/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0045 - root_mean_squared_error: 0.0671
Epoch 27/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0053 - root_mean_squared_error: 0.0727
Epoch 28/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0051 - root_mean_squared_error: 0.0711
Epoch 29/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0045 - root_mean_squared_error: 0.0670
Epoch 30/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0045 - root_mean_squared_error: 0.0667
Epoch 31/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0053 - root_mean_squared_error: 0.0725
Epoch 32/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0051 - root_mean_squared_error: 0.0711
Epoch 33/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0056 - root_mean_squared_error: 0.0750
Epoch 34/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0046 - root_mean_squared_error: 0.0675
Epoch 35/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0043 - root_mean_squared_error: 0.0654
Epoch 36/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0048 - root_mean_squared_error: 0.0691
Epoch 37/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0049 - root_mean_squared_error: 0.0698
Epoch 38/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0044 - root_mean_squared_error: 0.0666
Epoch 39/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0050 - root_mean_squared_error: 0.0706
Epoch 40/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0046 - root_mean_squared_error: 0.0675
Epoch 41/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0045 - root_mean_squared_error: 0.0674
Epoch 42/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0041 - root_mean_squared_error: 0.0640
Epoch 43/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0047 - root_mean_squared_error: 0.0686
Epoch 44/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0043 - root_mean_squared_error: 0.0653
Epoch 45/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0045 - root_mean_squared_error: 0.0669
Epoch 46/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0043 - root_mean_squared_error: 0.0658
Epoch 47/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0047 - root_mean_squared_error: 0.0688
Epoch 48/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0047 - root_mean_squared_error: 0.0684
Epoch 49/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0042 - root_mean_squared_error: 0.0648
Epoch 50/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0047 - root_mean_squared_error: 0.0682
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We rec
Model saved to models/abc_lstm_model.h5
Epoch 1/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 3s 10ms/step - loss: 0.0829 - root_mean_squared_error: 0.2787
Epoch 2/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0124 - root_mean_squared_error: 0.1112
Epoch 3/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0082 - root_mean_squared_error: 0.0906
Epoch 4/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0078 - root_mean_squared_error: 0.0882
Epoch 5/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0068 - root_mean_squared_error: 0.0823
Epoch 6/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0070 - root_mean_squared_error: 0.0835
Epoch 7/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0070 - root_mean_squared_error: 0.0837
Epoch 8/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0065 - root_mean_squared_error: 0.0806
Epoch 9/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0067 - root_mean_squared_error: 0.0817
Epoch 10/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0070 - root_mean_squared_error: 0.0836
Epoch 11/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 0.0062 - root_mean_squared_error: 0.0788
Epoch 12/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0064 - root_mean_squared_error: 0.0802
Epoch 13/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - loss: 0.0061 - root_mean_squared_error: 0.0778
Epoch 14/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0069 - root_mean_squared_error: 0.0831
Epoch 15/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0070 - root_mean_squared_error: 0.0833
Epoch 16/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0068 - root_mean_squared_error: 0.0824
Epoch 17/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0071 - root_mean_squared_error: 0.0839
Epoch 18/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0071 - root_mean_squared_error: 0.0844
Epoch 19/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0066 - root_mean_squared_error: 0.0814
Epoch 20/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0062 - root_mean_squared_error: 0.0786
Epoch 21/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0059 - root_mean_squared_error: 0.0770
Epoch 22/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0070 - root_mean_squared_error: 0.0837
Epoch 23/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0062 - root_mean_squared_error: 0.0789
Epoch 24/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0067 - root_mean_squared_error: 0.0817
Epoch 25/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0065 - root_mean_squared_error: 0.0806
Epoch 26/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0061 - root_mean_squared_error: 0.0784
Epoch 27/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0062 - root_mean_squared_error: 0.0786
Epoch 28/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0066 - root_mean_squared_error: 0.0811
Epoch 29/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0064 - root_mean_squared_error: 0.0800
Epoch 30/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0062 - root_mean_squared_error: 0.0788
Epoch 31/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0066 - root_mean_squared_error: 0.0811
Epoch 32/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0062 - root_mean_squared_error: 0.0784
Epoch 33/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0060 - root_mean_squared_error: 0.0776
Epoch 34/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0062 - root_mean_squared_error: 0.0786
Epoch 35/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0060 - root_mean_squared_error: 0.0776
Epoch 36/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0061 - root_mean_squared_error: 0.0783
Epoch 37/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0068 - root_mean_squared_error: 0.0824
Epoch 38/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0062 - root_mean_squared_error: 0.0787
Epoch 39/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0060 - root_mean_squared_error: 0.0774
Epoch 40/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0067 - root_mean_squared_error: 0.0819
Epoch 41/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0063 - root_mean_squared_error: 0.0794
Epoch 42/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0057 - root_mean_squared_error: 0.0757
Epoch 43/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0061 - root_mean_squared_error: 0.0783
Epoch 44/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0059 - root_mean_squared_error: 0.0770
Epoch 45/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0063 - root_mean_squared_error: 0.0791
Epoch 46/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0059 - root_mean_squared_error: 0.0770
Epoch 47/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0063 - root_mean_squared_error: 0.0792
Epoch 48/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0061 - root_mean_squared_error: 0.0781
Epoch 49/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0060 - root_mean_squared_error: 0.0773
Epoch 50/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0057 - root_mean_squared_error: 0.0752
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
Model saved to models/xyz_lstm_model.h5
model_preparation.py completed successfully.
Running model_testing.py...
2024-04-14 10:21:10.758668: I external/local_tsl/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-04-14 10:21:10.762901: I external/local_tsl/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-04-14 10:21:10.824441: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-04-14 10:21:12.019292: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
3/3 ━━━━━━━━━━━━━━━━━━━━ 1s 140ms/step
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.

Test result: MSE: 427.20374679058244, RMSE: 20.668907730951396

3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 121ms/step

Test result: MSE: 793.2677157831697, RMSE: 28.165008712641466
model_testing.py completed successfully.
All scripts completed successfully.
```

</details>

-------------------------
### [ДЗ №2](./lab2)

В рамках данного ДЗ было осуществлено следующее:
1. Развернут сервер с Jenkins, установлено необходимое программное обеспечение для работы над созданием модели машинного обучения.
2. Выбран способ получения данных – мы используем данные о кусре акций компании Walmart (`WMT`) с фондовой биржи, а также общеэкономические (макроэкономические данные). Источником данных выступает платформа https://www.alphavantage.co, с которой мы общаемся по API (для запуска скрипта сбора данных [lab2/gather_data.py](./lab2/gather_data.py) необходимо экспортировать переменную окружения `ALPHAVANTAGE_API_KEY`). Получить бесплатный API-ключ можно по ссылке: https://www.alphavantage.co/support/#api-key
3. Проведена обработка данных, выделены важные признаки, сформированы датасеты для тренировки и тестирования модели, сохранить – всё это выполняется в рамках скрипта [lab2/process_data.py](./lab2/process_data.py).
4. Создана и обучена на тренировочном датасете модель машинного обучения, сохранена в pickle или аналогичном формате. Данная логика реализована в [lab2/train_model.py](./lab2/train_model.py).
5. Модель, сохранённая на предыдущем этапе, загружена и проанализировано ее качество на тестовых данных – [lab2/test_model.py](./lab2/test_model.py).
6. Реализованы задания и конвейер. Конвейер связан с системой контроля версий. Конвейер сохранён как [lab2/Jenkinsfile](./lab2/Jenkinsfile).

<details>
  <summary>👈 Инструкция по запуску (проверялась на Debian 12):</summary>

```bash
mkdir test-lab-2

cd test-lab-2/

git clone https://github.com/khit-mle/mlops-practice.git .

python3 -m virtualenv .venv

source .venv/bin/activate

pip3 install -r requirements.txt

cd lab2/

python3 gather_data.py

python3 process_data.py

python3 train_model.py

python3 test_model.py

```
</details>

Скринкаст успешного запуска пайплайна на VPS с развёрнутым инстансом Jenkinks:
![lab2-jenkins]((./media/lab2/mlops_lab2_jenkins.gif))

