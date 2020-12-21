import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error


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

    def predict(self):
        try:
            # cria o modelo
            self.__create_model()
            # treina e valida o modelo
            history = self.__train()
            # testa o modelo
            loss, accuracy_model, rmse, mape, accuracy_detection = self.__test()
        except Exception as e:
            print('Falha ao realizar o predict! \n%s' % str(e))
