import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from pathlib import Path
matplotlib.use('Agg')


class MLP:
    """
    Classe que representa o modelo MLP
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
        self.input_dimension = input_dimension
        self.epochs = epochs
        self.layers = layers
        self.batch_size = 10
        self.data = data

    def __create_model(self):
        """
        Método responsável por realizar a criação do modelo
        """
        layers = self.layers

        # valida se foram informados pelo menos 3 camadas: entrada, intermediária 01 e saída
        if len(layers) < 3:
            raise Exception('ERRO! ')

        # instacia o modelo
        self.model = Sequential()

        # insere a camada de entrada
        input_layer = layers.pop(0)
        self.model.add(Dense(
            input_layer['qtd_neurons'],
            kernel_initializer='uniform',
            activation=input_layer['activation'],
            input_dim=self.input_dimension,
        ))

        # insere as camadas intermediárias e de saída
        for layer in layers:
            self.model.add(Dense(
                layer['qtd_neurons'],
                kernel_initializer='uniform',
                activation=layer['activation'],
            ))

        # Compilação do modelo com as métricas: R2, RMSE e MAPE
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', mean_squared_error, 'mape'],
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
            validation_data=(self.data['x_val'], self.data['y_val'])),
        return history

    def __test(self):
        """
        Método responsável por realizar o teste do modelo
        """
        # avalia o modelo com os dados de teste
        loss, accuracy_model, rmse, mape = self.model.evaluate(self.data['x_test'], self.data['y_test'])

        # gera as detecções se cada notícia é fake ou não
        detections = self.model.predict(self.data['x_data'])

        # ajusta as detecções
        rounded = [round(x[0]) for x in detections]
        accuracy_detection = np.mean(rounded == self.data['y_data'])

        return loss, accuracy_model, rmse, mape, accuracy_detection

    def __result(self, history, loss, accuracy_model, rmse, mape, accuracy_detection):
        """
        Método responsável por apresentar os resultados

        :param history: Histórico da detecção
        :type history: Tuple[Any]
        :param loss: Perca
        :type loss: float
        :param accuracy_model: Acurácia do modelo
        :type accuracy_model: float
        :param rmse: RMSE
        :type rmse: float
        :param mape: MAPE
        :type mape: float
        :param accuracy_detection: Acurácia da detecção
        :type accuracy_detection: ndarray
        """

        # Quanto menor a perda, mais próximas nossas previsões são dos rótulos verdadeiros.
        print('## Métricas')
        print('Loss: %.2f' % loss)
        print('R2: %.2f%%' % (accuracy_model * 100))
        print('R2 Detecções: %.2f%%' % (accuracy_detection * 100))
        print('MAPE: %.2f' % mape)
        print('RMSE: %.2f' % rmse)

        print('## Quantidades')
        print('QTD registros: %i ' % len(self.data['x']))
        print('QTD registros treino: %i ' % len(self.data['x_train']))
        print('QTD registros validação: %i ' % len(self.data['x_val']))
        print('QTD registros teste: %i ' % len(self.data['x_test']))
        print('QTD Épocas: %i' % self.epochs)

        # print('QTD neurônios camada de entrada: %i' % model_mlp['input_layer_quantity_neuron'])
        # print('QTD neurônios camadas intermediárias: %i' % model_mlp['hidden_layer_quantity_neuron'])
        # print('QTD de camadas intermediárias: %i' % model_mlp['hidden_layer_quantity'])
        # print('QTD de camadas intermediárias: %i' % model_mlp['hidden_layer_quantity'])
        #
        # print('## Funções de ativação ')
        # print('Camada de entrada: %s' % model_mlp['activation_function_input'])
        # print('Camada intermediária 01: %s' % model_mlp['activation_function_intermediary_01'])
        # print('Camada de saída: %s' % model_mlp['activation_function_output'])

        # Apresentação dos gráficos de treinamento e validação da rede
        Path('graphics').mkdir(parents=True, exist_ok=True)

        plt.plot(history.history['rmse'])
        plt.plot(history.history['val_rmse'])
        plt.title('RMSE - Treinamento e validação')
        plt.xlabel('Épocas')
        plt.ylabel('RMSE')
        plt.legend(['Treinamento', 'Validação'], loc='upper left')
        plt.savefig('graphics/mlp_rmse.png')
        plt.close()

        plt.plot(history.history['mape'])
        plt.plot(history.history['val_mape'])
        plt.title('MAPE')
        plt.xlabel('Épocas')
        plt.ylabel('MAPE')
        plt.legend(['Treinamento', 'Validação'], loc='upper left')
        plt.savefig('graphics/mlp_mape.png')
        plt.close()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('R2 - Treinamento e validação')
        plt.xlabel('Épocas')
        plt.ylabel('R2')
        plt.legend(['Treinamento', 'Validação'], loc='upper left')
        plt.savefig('graphics/mlp_r2.png')
        plt.close()

    def predict(self):
        """
        Método que realiza todas as etapas para detecção de Fake News com o modelo MLP
        """
        print('Iniciando a detecção de fake news com o modelo MLP... ')
        inicio = time.time()

        print('Criando o modelo... ')
        self.__create_model()

        print('Treinando e validando o modelo o modelo... ')
        history = self.__train()

        # print('Testando o modelo o modelo... ')
        # loss, accuracy_model, rmse, mape, accuracy_detection = self.__test()
        #
        # print('Resultados: ')
        # self.__result(history, loss, accuracy_model, rmse, mape, accuracy_detection)

        fim = time.time()
        print('Detecção de fake news realizada com sucesso! ')
        print('Tempo de execução: %f minutos' % ((fim - inicio) / 60))
