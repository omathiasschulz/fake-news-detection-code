import time
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def getData():
    """
    Método responsável por montar os dados fake para previsão
    """
    seq = np.array([
        # [x1, x2,  x3,  x4,  x5,  x6,   y],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5],
        [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.6],
        [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.7],
        [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.8],
        [0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.9],
    ])
    x, y = seq[:, :6], seq[:, 6]
    print(x)
    print(y)

    x = x.reshape(x.shape[0], x.shape[1], 1)
    print('x - shape: ' + str(x.shape))
    print(x)
    print('y - shape: ' + str(y.shape))
    print(y)

    return x, y


def info(y, previsao):
    """
    Método responsável por montar um gráfico comparando os resultados reais com os previstos

    :param y:
    :type y:
    :param previsao:
    :type previsao:
    :return:
    :rtype:
    """
    plt.plot(y, color='red', label='Real')
    plt.plot(previsao, color='blue', label='Previsto')
    plt.title('Previsões')
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.legend()
    plt.show()


def main():
    """
    Método main do script
    """
    print('Simples exemplo da utilização do modelo LSTM! ')
    inicio = time.time()

    # busca os dados
    x, y = getData()

    # cria o modelo
    model = Sequential()
    model.add(LSTM(10, input_shape=(x.shape[1], x.shape[2])))
    model.add(Dense(1, activation='linear'))

    # compila e fita o modelo
    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, epochs=10)

    # monta o predict e apresenta os resultados
    previsao = model.predict(x, verbose=0)
    print(previsao)
    info(y, previsao)

    fim = time.time()
    print('Tempo de execução: %.2f minutos' % ((fim - inicio) / 60))


if __name__ == '__main__':
    main()
