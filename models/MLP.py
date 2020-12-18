
class MLP():

    def teste(self, a):
        print('entrou %s' %a)

    
        # '''
        # Método responsável por realizar a criação do modelo MLP
        # '''
        # # Variável auxiliar - Quantidade de neurônio da camada de entrada
        # input_layer_quantity_neuron = 12
        # # Variável auxiliar - Quantidade de neurônio das camadas intermediárias
        # hidden_layer_quantity_neuron = 8
        # # Variável auxiliar - Quantidade de camadas intermediárias
        # hidden_layer_quantity = 1
        # # Variável auxiliar - Função de ativação da camada de entrada
        # activation_function_input = 'relu'
        # # Variável auxiliar - Função de ativação da camada intermediária 01
        # activation_function_intermediary_01 = 'relu'
        # # Variável auxiliar - Função de ativação da camada de saída
        # activation_function_output = 'sigmoid'

        # # Camada de entrada
        # model = Sequential()
        # model.add(Dense(
        #     input_layer_quantity_neuron, 
        #     input_dim = vector_dimension, 
        #     kernel_initializer = 'uniform', 
        #     activation = activation_function_input
        # ))

        # # Camada intermediária 01
        # model.add(Dense(
        #     hidden_layer_quantity_neuron, 
        #     kernel_initializer = 'uniform', 
        #     activation = activation_function_intermediary_01
        # ))

        # # Camada de saída
        # model.add(Dense(
        #     1, 
        #     kernel_initializer = 'uniform', 
        #     activation = activation_function_output
        # ))

        # # Compilação do modelo com as métricas: R2, RMSE e MAPE
        # model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', rmse, 'mape'])

        # return {
        #     'model': model,
        #     'input_layer_quantity_neuron': input_layer_quantity_neuron,
        #     'hidden_layer_quantity_neuron': hidden_layer_quantity_neuron,
        #     'hidden_layer_quantity': hidden_layer_quantity,
        #     'activation_function_input': activation_function_input,
        #     'activation_function_intermediary_01': activation_function_intermediary_01,
        #     'activation_function_output': activation_function_output,
        # }





class GeekforGeeks: 
  
    # default constructor 
    def __init__(self): 
        self.geek = "GeekforGeeks"
  
    # a method for printing data members 
    def print_Geek(self): 
        print(self.geek) 
  
  
# creating object of the class 
obj = GeekforGeeks() 
  
# calling the instance method using the object obj 
obj.print_Geek() 
