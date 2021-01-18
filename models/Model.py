import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from keras import backend
from sklearn.metrics import classification_report, confusion_matrix
matplotlib.use('Agg')
# setando um estilo padrão
sns.set_style('darkgrid')


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

    def __init__(self, model_name, info, data, path_graphics):
        """
        Construtor da classe

        :param model_name: Nome do modelo que foi instanciado
        :type model_name: str
        :param info: Dict com as informações epochs, batch_size e layers para montar o modelo
        :type info: dict
        :param data: Dados utilizados na detecção
        :type data: dict
        :param path_graphics: Diretório para salvar os gráficos gerados
        :type path_graphics: str
        """
        self.model = None
        self.model_name = model_name
        self.epochs = info['epochs']
        self.batch_size = info['batch_size'] if info.get('batch_size') else None
        self.layers = info['layers']
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
        predict = self.model.predict(self.data['x'])

        # valida a acurácia das detecções
        self.predictRounded = [round(x[0]) for x in predict]
        accuracy_detection = np.mean(self.predictRounded == self.data['y'])

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
        result = '\n# Teste ' + self.model_name + ' ' + str(datetime.now()) + '\n'
        result += 'camadas:'
        for key, layer in enumerate(self.layers):
            result += ' %i: ' % (key + 1)
            result += json.dumps(layer)

        result += '\nmétricas: '
        # quanto menor a perda, mais próximas nossas previsões são dos rótulos verdadeiros.
        result += 'loss: %.2f; ' % metrics['loss']
        result += 'accuracy_model(%%): %.2f; ' % (metrics['accuracy_model'] * 100)
        result += 'accuracy_detection(%%): %.2f; ' % (metrics['accuracy_detection'] * 100)
        result += 'mape: %.2f; ' % metrics['mape']
        result += 'rmse: %.2f; ' % metrics['rmse']

        cm = confusion_matrix(self.data['y'], self.predictRounded)
        result += 'confusion_matrix: '
        result += (str(cm[0, 0]) + '-' + str(cm[0, 1]) + '-' + str(cm[1, 0]) + '-' + str(cm[1, 1]) + ';')
        result += '\n'

        # apresenta os resultados e também salva no arquivo results.txt
        print(result)
        f = open('results/results.txt', 'a')
        f.write(result)
        f.close()

        # # Apresentação dos gráficos de treinamento e validação da rede
        # Path(self.path_graphics).mkdir(parents=True, exist_ok=True)
        #
        # sns_plot = sns.heatmap(cm, annot=True)
        # sns_plot.set_title('Matriz de Confusão')
        # sns_plot.set_xlabel('Valores Preditos')
        # sns_plot.set_ylabel('Valores Reais')
        # plt.savefig(self.path_graphics + 'cm.png')
        # plt.close()

        # plt.plot(history.history['rmseMetric'])
        # plt.plot(history.history['val_rmseMetric'])
        # plt.title(self.model_name + ' - RMSE')
        # plt.xlabel('Épocas')
        # plt.ylabel('RMSE')
        # plt.legend(['Treinamento', 'Validação'], loc='upper left')
        # plt.savefig(self.path_graphics + 'rmse.png')
        # plt.close()
        #
        # plt.plot(history.history['mape'])
        # plt.plot(history.history['val_mape'])
        # plt.title(self.model_name + ' - MAPE')
        # plt.xlabel('Épocas')
        # plt.ylabel('MAPE')
        # plt.legend(['Treinamento', 'Validação'], loc='upper left')
        # plt.savefig(self.path_graphics + 'mape.png')
        # plt.close()
        #
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title(self.model_name + ' - Acurácia')
        # plt.xlabel('Épocas')
        # plt.ylabel('Acurácia')
        # plt.legend(['Treinamento', 'Validação'], loc='upper left')
        # plt.savefig(self.path_graphics + 'acuracia.png')
        # plt.close()

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
