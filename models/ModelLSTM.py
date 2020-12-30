from keras.models import Sequential
from models.Model import Model
from keras.layers import Dense, LSTM, Dropout


class ModelLSTM(Model):
    """
    Classe que representa o modelo LSTM
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
        super().__init__('LSTM', info, data, 'graphics/lstm/')

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
        # if len(layers) < 3:
        #     raise Exception('Informe pelo menos três camadas! ')
        # instacia o modelo
        self.model = Sequential()

        # insere a camada de entrada
        input_layer = layers.pop(0)
        self.model.add(LSTM(
            input_layer['qtd_neurons'],
            kernel_initializer='uniform',
            activation=input_layer['activation'],
            # input_shape=(NUM_ENTRADAS, QTD_INFO_POR_ENTRADA)
            input_shape=(self.data['x_train'].shape[1], self.data['x_train'].shape[2]),
            return_sequences=True if input_layer.get('return_sequences') else False
        ))

        # insere as camadas intermediárias e de saída
        for layer in layers:
            # valida se é uma camada LSTM
            if layer['type'] == Model.LAYER_LSTM:
                self.model.add(LSTM(
                    layer['qtd_neurons'],
                    kernel_initializer='uniform',
                    activation=layer['activation'],
                    # input_shape=(NUM_ENTRADAS, QTD_INFO_POR_ENTRADA)
                    return_sequences=True if layer.get('return_sequences') else False
                ))
                continue
            # valida se é uma camada MLP
            if layer['type'] == Model.LAYER_MLP:
                self.model.add(Dense(
                    layer['qtd_neurons'],
                    kernel_initializer='uniform',
                    activation=layer['activation'],
                ))
                continue
            # valida se é uma camada de DROPOUT
            if layer['type'] == Model.LAYER_DROPOUT:
                self.model.add(Dropout(layer['value']))
                continue
            raise Exception('Camada inválida! ')
