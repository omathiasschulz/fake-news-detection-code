from models.Model import Model


class ModelLSTM(Model):
    """
    Classe que representa o modelo LSTM
    """

    def __init__(self, input_dimension, epochs, batch_size, layers, data):
        """
        Construtor da classe

        :param input_dimension: Quantidade de parâmetros de entrada para o modelo
        :type input_dimension: int
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
        super().__init__(input_dimension, epochs, batch_size, layers, data, 'graphics/lstm/')

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
