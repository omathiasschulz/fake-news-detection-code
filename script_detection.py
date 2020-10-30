import time, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from pathlib import Path
matplotlib.use('Agg')

VECTOR_DIMENSION = 300
EPOCHS = 150

def rmse(y_true, y_pred):
    '''
    Método responsável por realizar o cálculo do RMSE
    '''
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis = -1))

def createModelMLP(vector_dimension = 300):
    '''
    Método responsável por realizar a criação do modelo MLP
    '''
    # Variável auxiliar - Quantidade de neurônio da camada de entrada
    input_layer_quantity_neuron = 12
    # Variável auxiliar - Quantidade de neurônio das camadas intermediárias
    hidden_layer_quantity_neuron = 8
    # Variável auxiliar - Quantidade de camadas intermediárias
    hidden_layer_quantity = 1

    # Camada de entrada
    model = Sequential()
    model.add(Dense(
        input_layer_quantity_neuron, 
        input_dim = vector_dimension, 
        kernel_initializer = 'uniform', 
        activation = 'relu'
    ))

    # Camada intermediária 01
    model.add(Dense(
        hidden_layer_quantity_neuron, 
        kernel_initializer = 'uniform', 
        activation = 'relu'
    ))

    # Camada de saída
    model.add(Dense(
        1, 
        kernel_initializer = 'uniform', 
        activation = 'sigmoid'
    ))

    # Compilação do modelo com as métricas: R2, RMSE e MAPE
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', rmse, 'mape'])

    return model, input_layer_quantity_neuron, hidden_layer_quantity_neuron, hidden_layer_quantity

def train(model, x_train, y_train, x_val, y_val, epochs = 150):
    '''
    Método responsável por realizar o treinamento e validação do modelo
    '''
    history = model.fit(x_train, y_train, epochs = epochs, batch_size = 10, validation_data = (x_val, y_val))
    return model, history

def test(model, x_test, y_test, x_data, y_data):
    '''
    Método responsável por realizar o teste do modelo
    '''
    # Avalia o modelo com os dados de teste
    loss, accuracy_model, rmse, mape = model.evaluate(x_test, y_test)
    
    # Gera as detecções se cada notícia é fake ou não
    detections = model.predict(x_data)

    # Ajusta as detecções
    rounded = [round(x[0]) for x in detections]
    accuracy_detection = np.mean(rounded == y_data)

    return loss, accuracy_model, rmse, mape, accuracy_detection

def mlp(x_train, x_val, x_test, y_train, y_val, y_test):
    '''
    Método responsável por realizar a detecção de fake news com o modelo MLP
    '''
    model, input_layer_neuron, hidden_layer_neuron, hidden_layer_quantity = createModelMLP(VECTOR_DIMENSION)

    # Treinamento e validação do modelo
    model, history = train(model, x_train, y_train, x_val, y_val, EPOCHS)
    # Teste do modelo
    loss, accuracy_model, rmse, mape, accuracy_detection = test(model, x_test, y_test, x, y)

    print("\n\n### Resultados do teste da rede: ")
    # Quanto menor a perda, mais próximas nossas previsões são dos rótulos verdadeiros.
    print("Loss: %.2f" % loss)
    print("R2: %.2f%%" % (accuracy_model * 100))
    print("R2 Detecções: %.2f%%" % (accuracy_detection * 100))
    print("MAPE: %.2f" % mape)
    print("RMSE: %.2f" % rmse)
    print("###")
    print("QTD registros: %i " % len(x))
    print("QTD registros treino: %i " % len(x_train))
    print("QTD registros validação: %i " % len(x_val))
    print("QTD registros teste: %i " % len(x_test))
    print("QTD Épocas: %i" % EPOCHS)
    print("QTD neurônios camada de entrada: %i" % input_layer_neuron)
    print("QTD neurônios camadas intermediárias: %i" % hidden_layer_neuron)
    print("QTD de camadas intermediárias: %i" % hidden_layer_quantity)

    # Apresentação dos gráficos de treinamento e validação da rede
    Path('graphics').mkdir(parents=True, exist_ok=True)

    plt.plot(history.history['rmse'])
    plt.plot(history.history['val_rmse'])
    plt.title('RMSE - Treinamento e validação')
    plt.xlabel('Épocas')
    plt.ylabel('RMSE')
    plt.legend(['Treinamento', 'Validação'], loc='upper left')
    plt.savefig('graphics/mlp_rmse.png')

    plt.plot(history.history['mape'])
    plt.plot(history.history['val_mape'])
    plt.title('MAPE')
    plt.xlabel('Épocas')
    plt.ylabel('MAPE')
    plt.legend(['Treinamento', 'Validação'], loc='upper left')
    plt.savefig('graphics/mlp_mape.png')

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('R2 - Treinamento e validação')
    plt.xlabel('Épocas')
    plt.ylabel('R2')
    plt.legend(['Treinamento', 'Validação'], loc='upper left')
    plt.savefig('graphics/mlp_r2.png')

def lstm():
    # LSTM
    # https://www.kaggle.com/atishadhikari/fake-news-cleaning-word2vec-lstm-99-accuracy
    return True

try:
    print('Iniciando a construção dos modelos para detecção de fake news')
    inicio = time.time()

    # Realiza a leitura do CSV
    df = pd.read_csv('dataset_converted.csv', index_col=0)

    print('Dataset: ')
    print(df.head())

    # Realiza a separação do dataset entre X e Y
    y = df['fake_news'].to_numpy()
    df = df.drop(columns=['ID', 'fake_news'])    
    x = df.to_numpy()
 
    # Divisão dos dados
    # Treinamento => 70%
    # Validação => 20%
    # Teste => 10%
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size = 0.3)

    # MLP
    mlp(x_train, x_val, x_test, y_train, y_val, y_test)

    fim = time.time()
    print('Modelos para detecção de fake news criados com sucesso! ')
    print('Tempo de execução: %f minutos' %((fim - inicio) / 60))
except Exception as e:
    print('Falha ao gerar CSV: %s' %str(e))
