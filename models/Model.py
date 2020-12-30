import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from keras import backend
matplotlib.use('Agg')


def rmseMetric(y_true, y_pred):
    """
    Método responsável por realizar o cálculo do RMSE
    """
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


class Model:
    """
    Classe que representa um modelo genérico não funcional
    """

    # tipos de camadas
    LAYER_MLP = 'dense'
    LAYER_LSTM = 'lstm'
    LAYER_DROPOUT = 'dropout'

    # funções de ativação
    ATIVACAO_SIGMOID = 'sigmoid'
    ATIVACAO_TANH = 'tanh'
    ATIVACAO_RELU = 'relu'
    ATIVACAO_LEAKY_RELU = 'leaky relu'
    ATIVACAO_ELU = 'elu'

    def __init__(self, model_name, epochs, batch_size, layers, data, path_graphics):
        """
        Construtor da classe

        :param model_name: Nome do modelo que foi instanciado
        :type model_name: str
        :param epochs: Número de épocas para realizar o treinamento
        :type epochs: int
        :param batch_size: Número de exemplos de treinamento usados em uma iteração
        :type batch_size: int
        :param layers: Array de camadas que possui diferentes formatos de acordo com o modelo
        :type layers: list
        :param data: Dados utilizados na detecção
        :type data: dict
        :param path_graphics: Diretório para salvar os gráficos gerados
        :type path_graphics: str
        """
        self.model = None
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.layers = layers
        self.data = data
        self.path_graphics = path_graphics

    def _updateData(self):
        """
        Método para sobreescrever e realizar atualizações com os dados
        """
        return

    def _create_model(self):
        """
        Método para sobreescrever e realizar a construção do modelo
        """
        return

    def __compile(self):
        """
        Método responsável por compilar o modelo com as métricas: Acurácia, RMSE e MAPE
        """
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
        print('=> Modelo ' + self.model_name)
        for key, layer in enumerate(self.layers):
            print('camada_%i: ' % (key + 1), end='')
            print(layer)

        print('=> Métricas')
        # quanto menor a perda, mais próximas nossas previsões são dos rótulos verdadeiros.
        print('loss: %.2f; ' % metrics['loss'], end='')
        print('accuracy_model(%%): %.2f; ' % (metrics['accuracy_model'] * 100), end='')
        print('accuracy_detection(%%): %.2f; ' % (metrics['accuracy_detection'] * 100), end='')
        print('mape: %.2f; ' % metrics['mape'], end='')
        print('rmse: %.2f; \n' % metrics['rmse'])

        # Apresentação dos gráficos de treinamento e validação da rede
        Path(self.path_graphics).mkdir(parents=True, exist_ok=True)

        plt.plot(history.history['rmseMetric'])
        plt.plot(history.history['val_rmseMetric'])
        plt.title(self.model_name + ' - RMSE')
        plt.xlabel('Épocas')
        plt.ylabel('RMSE')
        plt.legend(['Treinamento', 'Validação'], loc='upper left')
        plt.savefig(self.path_graphics + 'rmse.png')
        plt.close()

        plt.plot(history.history['mape'])
        plt.plot(history.history['val_mape'])
        plt.title(self.model_name + ' - MAPE')
        plt.xlabel('Épocas')
        plt.ylabel('MAPE')
        plt.legend(['Treinamento', 'Validação'], loc='upper left')
        plt.savefig(self.path_graphics + 'mape.png')
        plt.close()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(self.model_name + ' - Acurácia')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.legend(['Treinamento', 'Validação'], loc='upper left')
        plt.savefig(self.path_graphics + 'acuracia.png')
        plt.close()

    def predict(self):
        """
        Método que realiza todas as etapas para detecção de Fake News com o modelo
        """
        try:
            print('Criando o modelo %s... ' % self.model_name)
            self._updateData()
            self._create_model()
            self.__compile()

            print('Treinando e validando o modelo... ')
            history = self.__train()

            print('Testando o modelo... ')
            metrics = self.__test()

            print('Resultados: ')
            self.__result(history, metrics)
        except Exception as e:
            print('Erro! %s' % str(e))
