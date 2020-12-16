# fake-news-detection-code

Detecção de Fake News utilizando os modelos de Redes Neurais Artificiais MLP (Multilayer perceptron) e LSTM (Long short-term memory)

## Instalar as dependências

**Obs:** Para rodar os scripts utilize o Python 3

Para instalar as dependências do projeto, na pasta raiz do projeto digite:

`pip3 install -r requirements.txt`

## Datasets

O projeto possui dois datasets, apresentados abaixo:

### Dataset em formato de texto

O `dataset.csv` é um dataset com as notícias em formato de texto

O dataset está disponível em [fake-new-dataset](https://github.com/mathiasarturschulz/fake-news-dataset)

### Dataset em formato numérico

O `dataset_converted.csv` é um dataset com as notícias convertidas para representação numérica gerado a partir do script `script_01_convert.py`

Os modelos de Redes Neurais Artificiais apenas trabalham com números, por isso deve ser realizado a conversão

O conversão é realizada com o auxílio da biblioteca [Gensim](https://radimrehurek.com/gensim/) utilizando o modelo Doc2Vec

Para criar o dataset, na pasta raiz do projeto digite:

`python3 script_01_convert.py`

**Resultados**

Tempo de execução: 1.10 minutos

## Detection

The detection of false and true news is carried out using script `script_detection.py`

To create the detection, in the project folder type:

`python3 script_detection.py`

**Time to generate the detection:**

Runtime: XXXX minutes

The script also creates the images in the `grapihcs` folder, the images are graphics of the quality detection metrics of the model detection
