import time
import pandas as pd
from models.MLP import MLP
from models.ModelLSTM import ModelLSTM
from sklearn.model_selection import train_test_split

VECTOR_DIMENSION = 300
EPOCHS = 150

print('Iniciando a construção dos modelos para detecção de fake news')
inicio = time.time()

# realiza a leitura do CSV
df = pd.read_csv('dataset_converted.csv', index_col=0)

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
print('Quantidade de épocas: %i' % EPOCHS)

# monta o modelo MLP
layers = [
    # camada de entrada
    {
        'qtd_neurons': 12,
        'activation': MLP.RELU,
    },
    # camada intermediária 01
    {
        'qtd_neurons': 8,
        'activation': MLP.RELU,
    },
    # camada de saída
    {
        'qtd_neurons': 1,
        'activation': MLP.SIGMOID,
    },
]

# # cria o modelo e inicia a detecção
# model_mlp = MLP(VECTOR_DIMENSION, EPOCHS, layers, data)
# model_mlp.predict()

# cria o modelo e inicia a detecção
model_lstm = ModelLSTM(VECTOR_DIMENSION, EPOCHS, layers, data)
model_lstm.predict()

fim = time.time()
print('Modelos para detecção de fake news criados com sucesso! ')
print('Tempo de execução: %.2f minutos' % ((fim - inicio) / 60))
