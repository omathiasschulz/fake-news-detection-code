from models.MLP import MLP
import time, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from pathlib import Path
matplotlib.use('Agg')

VECTOR_DIMENSION = 300
EPOCHS = 150

try:
    print('Iniciando a construção dos modelos para detecção de fake news')
    inicio = time.time()

    # Realiza a leitura do CSV
    df = pd.read_csv('dataset_converted.csv', index_col=0)

    print('Dataset: ')
    print(df.head())

    # Realiza a separação do dataset entre X e Y
    y = df['fake_news'].to_numpy()
    df = df.drop(columns=['ID', 'fake_news'])
    x = df.to_numpy()

    # Divisão dos dados
    # Treinamento => 70%
    # Validação => 20%
    # Teste => 10%
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.3)

    # monta o modelo MLP
    layers = [
        # camada de entrada
        {
            'qtd_neurons': 12,
            'activation': MLP.RELU,
        },
        # camada intermediária 01
        {
            'qtd_neurons': 8,
            'activation': MLP.RELU,
        },
        # camada de saída
        {
            'qtd_neurons': 1,
            'activation': MLP.SIGMOID,
        },
    ]

    data = {
        'x': x,
        'x_train': x_train,
        'x_val': x_val,
        'x_test': x_test,
        'y': y,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
    }
    print(data)
    model_mlp = MLP(VECTOR_DIMENSION, EPOCHS, layers, data)

    model_mlp.predict()

    fim = time.time()
    print('Modelos para detecção de fake news criados com sucesso! ')
    print('Tempo de execução: %f minutos' % ((fim - inicio) / 60))
except Exception as e:
    print('Falha ao gerar CSV: %s' % str(e))
