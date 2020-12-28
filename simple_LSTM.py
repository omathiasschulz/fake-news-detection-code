import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array
import matplotlib.pyplot as plt


def getData():
    """
    Método responsável por montar os dados fake para previsão
    """
    seq = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]
    seq = array(seq)
    x, y = seq[:, 0], seq[:, 1]
    x = x.reshape((len(x), 1, 1))

    print('x')
    print(x.shape)
    print(x)
    print('Y')
    print(y.shape)
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

    # cria o modelo
    model = Sequential()
    model.add(LSTM(10, input_shape=(1, 1)))
    model.add(Dense(1, activation='linear'))

    # busca os dados
    x, y = getData()

    # compila e fita o modelo
    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, epochs=300)

    # monta o predict e apresenta os resultados
    previsao = model.predict(x, verbose=0)
    print(previsao)
    info(y, previsao)

    fim = time.time()
    print('Tempo de execução: %.2f minutos' % ((fim - inicio) / 60))


if __name__ == '__main__':
    main()
