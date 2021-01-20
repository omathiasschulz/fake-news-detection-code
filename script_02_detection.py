import time
import pandas as pd
from models.Model import Model
from models.ModelMLP import ModelMLP
from models.ModelLSTM import ModelLSTM
from sklearn.model_selection import train_test_split


def generateData(csv_name):
    """
    Busca os dados e organiza para utilização nos modelos

    :return: Retorna um dict com os dados
    :rtype: dict
    """
    # realiza a leitura do CSV
    df = pd.read_csv(csv_name, index_col=0)
    print('Dataset: ')
    print(df.head())

    # realiza a separação do dataset entre X e Y
    y = df['fake_news'].to_numpy()
    df = df.drop(columns=['ID', 'fake_news'])
    x = df.to_numpy()

    # divisão dos dados
    # treinamento => 70% | validação => 20% | teste => 10%
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.3)

    # agrupa os dados em um dicionário
    data = {
        'x': x,
        'x_train': x_train,
        'x_val': x_val,
        'x_test': x_test,
        'y': y,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
    }
    print('Quantidade de registros: %i ' % len(x))
    print('Quantidade de registros para treino: %i ' % len(x_train))
    print('Quantidade de registros para validação: %i ' % len(x_val))
    print('Quantidade de registros para teste: %i ' % len(x_test))
    return data


def generateMLP(data):
    """
    Realiza a detecção de fake news com o modelo MLP

    :param data: Dados utilizados no modelo
    :type data: dict
    """
    model = {
        'epochs': 50,
        'batch_size': 10,
        'layers': [
            # camada de entrada
            {
                'qtd_neurons': 12,
                'activation': Model.ATIVACAO_RELU,
            },
            # camada intermediária 01
            {
                'qtd_neurons': 8,
                'activation': Model.ATIVACAO_RELU,
            },
            # camada de saída
            {
                'qtd_neurons': 1,
                'activation': Model.ATIVACAO_SIGMOID,
            },
        ]
    }
    model_mlp = ModelMLP(model, data)
    model_mlp.predict()


def generateLSTM(data):
    """
    Realiza a detecção de fake news com o modelo LSTM

    :param data: Dados utilizados no modelo
    :type data: dict
    """
    model = {
        'epochs': 20,
        'layers': [
            # camada de entrada
            {
                # a primeira camada sempre é LSTM
                'qtd_neurons': 12,
                'activation': Model.ATIVACAO_RELU,
                'return_sequences': True,
            },
            # camada intermediária 01
            {
                'type': Model.LAYER_DROPOUT,
                'value': 0.2,
            },
            # camada intermediária 02
            {
                'type': Model.LAYER_LSTM,
                'qtd_neurons': 8,
                'activation': Model.ATIVACAO_RELU,
            },
            # camada de saída
            {
                'type': Model.LAYER_MLP,
                'qtd_neurons': 1,
                'activation': Model.ATIVACAO_SIGMOID,
            },
        ]
    }
    model_lstm = ModelLSTM(model, data)
    model_lstm.predict()


# TEXT_LENGTH = [50, 100, 150, 200]
# utilizando apenas um dataset para testes
TEXT_LENGTH = [50]
PATH_DATASETS_FORMATTED = 'datasets/formatted/'
PATH_DATASETS_CONVERTED = 'datasets/converted/'


def main():
    """
    Método main do script
    """
    print('Iniciando a detecção de fake news')
    inicio = time.time()

    # realiza um teste de um mesmo modelo nos 4 datasets com tamanho dos textos variados
    for dataset_atual in TEXT_LENGTH:
        dataset_nome = PATH_DATASETS_CONVERTED + 'dataset_%i_palavras.csv' % dataset_atual
        print('\n\n' + dataset_nome)
        data = generateData(dataset_nome)
        generateMLP(data)
        # generateLSTM(data)

    fim = time.time()
    print('Detecção de fake news realizada com sucesso! ')
    print('Tempo de execução: %.2f minutos' % ((fim - inicio) / 60))


if __name__ == '__main__':
    main()
