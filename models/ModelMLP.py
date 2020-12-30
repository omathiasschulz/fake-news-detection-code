from keras.models import Sequential
from models.Model import Model
from keras.layers import Dense


class ModelMLP(Model):
    """
    Classe que representa o modelo MLP
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
        super().__init__('MLP', epochs, batch_size, layers, data, 'graphics/mlp/')

    def _create_model(self):
        """
        Método responsável por realizar a criação do modelo
        """
        # passa as camadas para uma nova variável sem ser por referência
        layers = self.layers[:]

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
            input_dim=self.data['x_train'].shape[1],
        ))

        # insere as camadas intermediárias e de saída
        for layer in layers:
            self.model.add(Dense(
                layer['qtd_neurons'],
                kernel_initializer='uniform',
                activation=layer['activation'],
            ))
