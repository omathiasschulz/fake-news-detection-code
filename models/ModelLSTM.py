from models.Model import Model


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

    def predict(self):
        """
        Método que realiza todas as etapas para detecção de Fake News com o modelo
        """
        print('\n=> Modelo LSTM')
        super().predict()
