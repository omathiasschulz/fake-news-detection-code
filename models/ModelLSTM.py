import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from keras import backend
from models.Model import Model
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from pathlib import Path
from keras import backend
from models.Model import Model, rmseMetric
matplotlib.use('Agg')



class ModelLSTM(Model):
    """
    Classe que representa o modelo LSTM
    """

    def __init__(self, epochs, batch_size, layers, data):
        """
        Construtor da classe

        :param epochs: Número de épocas para realizar o treinamento
        :type epochs: int
        :param batch_size: Número de exemplos de treinamento usados em uma iteração
        :type batch_size: int
        :param layers: Camadas para montar o modelo no formato:
            [{'qtd_neurons': qtd_neuronios, 'activation': 'funcao_ativacao',}]
        :type layers: list
        :param data: Dados utilizados na detecção
        :type data: dict
        """
        super().__init__('LSTM', epochs, batch_size, layers, data, 'graphics/lstm/')

    def _updateData(self):
        """
        Método sobreescrito para realizar atualizações com o dataset
        """
        # realiza o reshape dos dados para serem utilizados no modelo LSTM
        self.data['x'] = self.data['x'].reshape(
            self.data['x'].shape[0], self.data['x'].shape[1], 1
        )
        self.data['x_train'] = self.data['x_train'].reshape(
            self.data['x_train'].shape[0], self.data['x_train'].shape[1], 1
        )
        self.data['x_test'] = self.data['x_test'].reshape(
            self.data['x_test'].shape[0], self.data['x_test'].shape[1], 1
        )
        self.data['x_val'] = self.data['x_val'].reshape(
            self.data['x_val'].shape[0], self.data['x_val'].shape[1], 1
        )

    def _create_model(self):
        """
        Método responsável por realizar a criação do modelo
        """
        # passa as camadas para uma nova variável sem ser por referência
        layers = self.layers[:]
        # valida se foram informados pelo menos 3 camadas: entrada, intermediária 01 e saída
        if len(layers) < 3:
            raise Exception('Informe pelo menos três camadas! ')
        # instacia o modelo
        self.model = Sequential()

        # insere a camada de entrada
        input_layer = layers.pop(0)
        if input_layer['type'] == self.LAYER_MLP:
            self.model.add(Dense(
                input_layer['qtd_neurons'],
                kernel_initializer='uniform',
                activation=input_layer['activation'],
                input_dim=self.data['x_train'].shape[1],
            ))
        elif input_layer['type'] == self.LAYER_LSTM:
            print('return_sequences')
            print(True if input_layer['return_sequences'] else False)
            self.model.add(LSTM(
                input_layer['qtd_neurons'],
                kernel_initializer='uniform',
                activation=input_layer['activation'],
                # input_shape=(NUM_ENTRADAS, QTD_INFO_POR_ENTRADA)
                input_shape=(self.data['x_train'].shape[1], self.data['x_train'].shape[2]),
                return_sequences=True if input_layer['return_sequences'] else False
            ))
        else:
            raise Exception('Camada de entrada inválida! ')

        # insere as camadas intermediárias e de saída
        for layer in layers:
            self.model.add(Dense(
                layer['qtd_neurons'],
                kernel_initializer='uniform',
                activation=layer['activation'],
            ))
