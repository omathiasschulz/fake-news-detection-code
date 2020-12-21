from models.MLP import MLP

VECTOR_DIMENSION = 300
EPOCHS = 150

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

# model_mlp = MLP(VECTOR_DIMENSION, EPOCHS, layers)

data = {
    x,
    x_train,
    x_val,
    x_test,
    y,
    y_train,
    y_val,
    y_test,
}

# model_mlp.predict()
