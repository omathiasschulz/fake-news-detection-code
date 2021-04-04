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


def generateMLP(data, dataset_name):
    """
    Realiza a detecção de fake news com o modelo MLP

    :param data: Dados utilizados no modelo
    :type data: dict
    :param dataset_name: Nome do dataset utilizado
    :type dataset_name: str
    """
    # MELHOR MODELO MLP NO DATASET DE 50 PALAVRAS
    model = {
        "epochs": 50,
        "batch_size": 2,
        "layers": [
            {"qtd_neurons": 300, "activation": "relu"},
            {"qtd_neurons": 128, "activation": "tanh"},
            {"qtd_neurons": 1, "activation": "sigmoid"}
        ]
    }
    model_mlp = ModelMLP(model, data, dataset_name)
    model_mlp.predict()


def generateLSTM(data, dataset_name):
    """
    Realiza a detecção de fake news com o modelo LSTM

    :param data: Dados utilizados no modelo
    :type data: dict
    :param dataset_name: Nome do dataset utilizado
    :type dataset_name: str
    """
    # MELHOR MODELO LSTM NO DATASET DE 50 PALAVRAS
    model = {
        "epochs": 50,
        "batch_size": 32,
        "layers": [
            {"qtd_neurons": 300, "activation": "elu", "return_sequences": True},
            {"type": "lstm", "qtd_neurons": 90, "activation": "tanh"},
            {"type": "dense", "qtd_neurons": 1, "activation": "sigmoid"}
        ]
    }
    model_lstm = ModelLSTM(model, data, dataset_name)
    model_lstm.predict()


PATH_DATASETS_FORMATTED = 'datasets/formatted/'
PATH_DATASETS_CONVERTED = 'datasets/converted/'


def main():
    """
    Método main do script
    """
    print('Iniciando a detecção de fake news')
    inicio = time.time()

    # testes com o dataset de 50 palavras no modelo MLP
    text_length = 50
    dataset_nome = PATH_DATASETS_CONVERTED + 'dataset_%i_palavras.csv' % text_length
    print('\n\n' + dataset_nome)
    data = generateData(dataset_nome)
    generateMLP(data, 'dataset_%i_palavras.csv' % text_length)

    # testes com o dataset de 50 palavras no modelo LSTM
    text_length = 50
    dataset_nome = PATH_DATASETS_CONVERTED + 'dataset_%i_palavras.csv' % text_length
    print('\n\n' + dataset_nome)
    data = generateData(dataset_nome)
    generateLSTM(data, 'dataset_%i_palavras.csv' % text_length)

    fim = time.time()
    print('Detecção de fake news realizada com sucesso! ')
    print('Tempo de execução: %.2f minutos' % ((fim - inicio) / 60))


if __name__ == '__main__':
    main()
