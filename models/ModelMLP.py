from keras.models import Sequential
from models.Model import Model
from keras.layers import Dense


class ModelMLP(Model):
    """
    Classe que representa o modelo MLP
    """

    def __init__(self, info, data):
        """
        Construtor da classe

        :param info: Dict com as informações epochs, batch_size e layers para montar o modelo
            As camadas para montar o modelo estão no formato:
            [{'qtd_neurons': qtd_neuronios, 'activation': 'funcao_ativacao',}]
        :type info: dict
        :param data: Dados utilizados na detecção
        :type data: dict
        """
        super().__init__('MLP', info, data, 'graphics/mlp/')

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
