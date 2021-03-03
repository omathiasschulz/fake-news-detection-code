import json
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from keras import backend
from sklearn.metrics import confusion_matrix, classification_report
matplotlib.use('Agg')
# setando um estilo padrão
sns.set_style('darkgrid')


def rmse(y_true, y_pred):
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
    ATIVACAO_ELU = 'elu'

    def __init__(self, model_name, info, data, dataset_name, path_graphics):
        """
        Construtor da classe

        :param model_name: Nome do modelo que foi instanciado
        :type model_name: str
        :param info: Dict com as informações epochs, batch_size e layers para montar o modelo
        :type info: dict
        :param data: Dados utilizados na detecção
        :type data: dict
        :param dataset_name: Nome do dataset utilizado
        :type dataset_name: str
        :param path_graphics: Diretório para salvar os gráficos gerados
        :type path_graphics: str
        """
        self.model = None
        self.model_name = model_name
        self.info = info
        self.epochs = info['epochs']
        self.batch_size = info['batch_size'] if info.get('batch_size') else None
        self.layers = info['layers']
        self.data = data
        self.dataset_name = dataset_name
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
        Método responsável por compilar o modelo com as métricas: Acurácia e RMSE
        """
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy', rmse],
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
        m_loss, m_accuracy_model, m_rmse = self.model.evaluate(self.data['x_test'], self.data['y_test'])

        # gera as detecções se cada notícia é fake ou não
        predict = self.model.predict(self.data['x'])

        # valida a acurácia das detecções
        self.predictRounded = [round(x[0]) for x in predict]
        m_accuracy_detection = np.mean(self.predictRounded == self.data['y'])

        # monta um dict das métricas e retorna
        return {
            'loss': m_loss,
            'accuracy_model': m_accuracy_model,
            'accuracy_detection': m_accuracy_detection,
            'rmse': m_rmse,
        }

    def __result(self, history, metrics):
        """
        Método responsável por apresentar os resultados

        :param history: Histórico das métricas da detecção
        :type history: Tuple[Any]
        :param metrics: Resultado obtido pelas métricas
        :type metrics: dict
        """
        result = '\n# Teste ' + self.model_name + ' - ' + str(datetime.now()) + ' - ' + self.dataset_name + '\n'
        result += json.dumps(self.info)

        result += '\nmétricas: '
        # quanto menor a perda, mais próximas nossas previsões são dos rótulos verdadeiros.
        result += 'loss: %.4f; ' % metrics['loss']
        result += 'accuracy_model(%%): %.4f; ' % (metrics['accuracy_model'] * 100)
        result += 'accuracy_detection(%%): %.4f; ' % (metrics['accuracy_detection'] * 100)
        result += 'rmse: %.4f; ' % metrics['rmse']

        report = classification_report(self.data['y'], self.predictRounded, output_dict=True)
        # macro avg dá a cada previsão um peso semelhante ao calcular a perda
        result += 'precision: %.4f; ' % report['macro avg']['precision']
        result += 'recall: %.4f; ' % report['macro avg']['recall']
        result += 'f1-score: %.4f; ' % report['macro avg']['f1-score']

        cm = confusion_matrix(self.data['y'], self.predictRounded)
        result += 'confusion_matrix: '
        result += (str(cm[0, 0]) + '-' + str(cm[0, 1]) + '-' + str(cm[1, 0]) + '-' + str(cm[1, 1]) + ';')
        result += '\n'

        # apresenta os resultados e também salva no arquivo results.txt
        print(result)
        f = open('results/results.txt', 'a')
        f.write(result)
        f.close()

        # gera os gráficos
        self.__graphics(history, cm)
        # salva o modelo
        self.__save_model()

    def __graphics(self, history, cm):
        """
        Método responsável por gerar os gráficos

        :param history: Histórico das métricas da detecção
        :type history: Tuple[Any]
        :param cm: Matriz de confusão
        :type cm: list
        """
        # Apresentação dos gráficos de treinamento e validação da rede
        Path(self.path_graphics).mkdir(parents=True, exist_ok=True)

        # cm
        cm = pd.DataFrame(cm, index=['Fake', 'Original'], columns=['Fake', 'Original'])
        sns_plot = sns.heatmap(
            cm, linecolor='black', linewidth=1, annot=True, fmt='',
            xticklabels=['Fake', 'Original'], yticklabels=['Fake', 'Original']
        )
        sns_plot.set_title(self.model_name + ' - Matriz de Confusão')
        sns_plot.set_xlabel('Valores Preditos')
        sns_plot.set_ylabel('Valores Reais')
        plt.savefig(self.path_graphics + 'cm.png', dpi=300)
        plt.close()

        # rmse
        plt.plot(history.history['rmse'], 'go-', markersize=3, label='Treinamento')
        plt.plot(history.history['val_rmse'], 'ro-', markersize=3, label='Validação')
        plt.title(self.model_name + ' - RMSE')
        plt.xlabel('Épocas')
        plt.ylabel('RMSE')
        plt.legend()
        plt.savefig(self.path_graphics + 'rmse.png', dpi=300)
        plt.close()

        # acurácia
        plt.plot(history.history['accuracy'], 'go-', markersize=3, label='Treinamento')
        plt.plot(history.history['val_accuracy'], 'ro-', markersize=3, label='Validação')
        plt.title(self.model_name + ' - Acurácia')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.savefig(self.path_graphics + 'acuracia.png', dpi=300)
        plt.close()

        # loss
        plt.plot(history.history['loss'], 'go-', markersize=3, label='Treinamento')
        plt.plot(history.history['val_loss'], 'ro-', markersize=3, label='Validação')
        plt.title(self.model_name + ' - Loss')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.path_graphics + 'loss.png', dpi=300)
        plt.close()

    def __save_model(self):
        """
        Método responsável por salvar o modelo
        """
        self.model.save('results/modelo_mlp.h5')

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
