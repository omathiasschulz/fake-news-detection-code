import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from pathlib import Path
from keras import backend
matplotlib.use('Agg')


def rmseMetric(y_true, y_pred):
    """
    Método responsável por realizar o cálculo do RMSE
    """
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


class ModelLSTM:
    """
    Classe que representa o modelo LSTM
    """

    # constantes das funções de ativação
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    RELU = 'relu'
    LEAKY_RELU = 'leaky relu'
    ELU = 'elu'

    def __init__(self, input_dimension, epochs, layers, data):
        """
        Construtor da classe

        :param input_dimension: Quantidade de parâmetros de entrada para o modelo
        :type input_dimension: int
        :param epochs: Número de épocas para realizar o treinamento
        :type epochs: int
        :param layers: Camadas para montar o modelo no formato:
            [{'qtd_neurons': qtd_neuronios, 'activation': 'funcao_ativacao',}]
        :type layers: list
        :param data: Dados utilizados na detecção
        :type data: dict
        """
        self.path_graphics = 'graphics/lstm/'
        self.input_dimension = input_dimension
        self.epochs = epochs
        self.layers = layers
        self.batch_size = 10
        self.data = data

    def __create_model(self):
        """
        Método responsável por realizar a criação do modelo
        """
        # passa as camadas para uma nova variável sem ser por referência
        layers = self.layers[:]

        # valida se foram informados pelo menos 3 camadas: entrada, intermediária 01 e saída
        if len(layers) < 3:
            raise Exception('ERRO! ')

        # realiza o reshape dos dados para serem utilizados no modelo LSTM
        self.data['x_train'] = self.data['x_train'].reshape(
            self.data['x_train'].shape[0], self.data['x_train'].shape[1], 1
        )
        self.data['x_val'] = self.data['x_val'].reshape(
            self.data['x_val'].shape[0], self.data['x_val'].shape[1], 1
        )
        self.data['x_test'] = self.data['x_test'].reshape(
            self.data['x_test'].shape[0], self.data['x_test'].shape[1], 1
        )
        self.data['x'] = self.data['x'].reshape(
            self.data['x'].shape[0], self.data['x'].shape[1], 1
        )

        # instacia o modelo
        self.model = Sequential()

        self.model.add(LSTM(4, input_shape=(self.data['x_train'].shape[1], self.data['x_train'].shape[2])))

        # compilação do modelo com as métricas: R2, RMSE e MAPE
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', rmseMetric, 'mape'],
        )

    def __train(self):
        """
        Método responsável por realizar o treinamento e validação do modelo
        """
        history = self.model.fit(
            self.data['x_train'],
            self.data['y_train'],
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.data['x_val'], self.data['y_val'])
        )

        return history

    def __test(self):
        """
        Método responsável por realizar o teste do modelo
        """
        # avalia o modelo com os dados de teste
        loss, accuracy_model, rmse, mape = self.model.evaluate(self.data['x_test'], self.data['y_test'])

        # gera as detecções se cada notícia é fake ou não
        detections = self.model.predict(self.data['x'])

        # valida a acurácia das detecções
        rounded = [round(x[0]) for x in detections]
        accuracy_detection = np.mean(rounded == self.data['y'])

        # monta um dict das métricas e retorna
        return {
            'loss': loss,
            'accuracy_model': accuracy_model,
            'accuracy_detection': accuracy_detection,
            'rmse': rmse,
            'mape': mape,
        }

    def __result(self, history, metrics):
        """
        Método responsável por apresentar os resultados

        :param history: Histórico das métricas da detecção
        :type history: Tuple[Any]
        :param metrics: Resultado obtido pelas métricas
        :type metrics: dict
        """
        print('=> Modelo LSTM')
        for key, layer in enumerate(self.layers):
            print('Camada %i: ' % (key + 1), end='')
            print('qtd_neurons: %i; ' % layer['qtd_neurons'], end='')
            print('activation_fn: %s; ' % layer['activation'])

        print('=> Métricas')
        # quanto menor a perda, mais próximas nossas previsões são dos rótulos verdadeiros.
        print('loss: %.2f; ' % metrics['loss'], end='')
        print('R2_model: %.2f%%; ' % (metrics['accuracy_model'] * 100), end='')
        print('R2_detection: %.2f%%; ' % (metrics['accuracy_detection'] * 100), end='')
        print('MAPE: %.2f; ' % metrics['mape'], end='')
        print('RMSE: %.2f; \n' % metrics['rmse'])

        # Apresentação dos gráficos de treinamento e validação da rede
        Path(self.path_graphics).mkdir(parents=True, exist_ok=True)

        plt.plot(history.history['rmseMetric'])
        plt.plot(history.history['val_rmseMetric'])
        plt.title('RMSE - Treinamento e validação')
        plt.xlabel('Épocas')
        plt.ylabel('RMSE')
        plt.legend(['Treinamento', 'Validação'], loc='upper left')
        plt.savefig(self.path_graphics + 'rmse.png')
        plt.close()

        plt.plot(history.history['mape'])
        plt.plot(history.history['val_mape'])
        plt.title('MAPE')
        plt.xlabel('Épocas')
        plt.ylabel('MAPE')
        plt.legend(['Treinamento', 'Validação'], loc='upper left')
        plt.savefig(self.path_graphics + 'mape.png')
        plt.close()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('R2 - Treinamento e validação')
        plt.xlabel('Épocas')
        plt.ylabel('R2')
        plt.legend(['Treinamento', 'Validação'], loc='upper left')
        plt.savefig(self.path_graphics + 'r2.png')
        plt.close()

    def predict(self):
        """
        Método que realiza todas as etapas para detecção de Fake News com o modelo
        """
        print('Iniciando a detecção de fake news com o modelo LSTM... ')

        print('Criando o modelo... ')
        self.__create_model()

        print('Treinando e validando o modelo o modelo... ')
        history = self.__train()

        print('Testando o modelo o modelo... ')
        metrics = self.__test()

        print('Detecção de fake news realizada com sucesso! ')
        print('\nResultados: ')
        self.__result(history, metrics)
