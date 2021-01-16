# fake-news-detection-code :newspaper:

Detecção de Fake News utilizando os modelos de Redes Neurais Artificiais MLP (Multilayer perceptron) e LSTM (Long short-term memory)

## Instalar as dependências

**Obs:** Para rodar os scripts utilize o Python 3.8

Para instalar as dependências do projeto, na pasta raiz do projeto digite:

`pip3 install -r requirements.txt`

## Datasets

O projeto possui dois tipos de datasets, apresentados abaixo:

### Datasets em formato de texto

Os datasets que estão na pasta `datasets/formatted/` possuem as notícias em formato de texto

Os datasets estão disponíveis em [fake-new-dataset](https://github.com/mathiasarturschulz/fake-news-dataset)

Os datasets dessa pasta foram criados com base na variável TEXT_LENGTH, no qual o valor determina que o dataset não deve possuir notícias menores que o valor especificado

Por exemplo: O dataset do CSV dataset_100_palavras.csv possui apenas textos maiores ou iguais que 100 palavras e o dataset do CSV dataset_200_palavras.csv possui apenas textos maiores ou iguais que 200 palavras

### Datasets em formato numérico

Os datasets que estão na pasta `datasets/converted/` possuem as notícias convertidas para representação numérica gerados a partir do script `script_01_convert.py`

Os datasets com os textos convertidos dessa pasta foram criados com base na variável TEXT_LENGTH, no qual o valor determina que o dataset possui o tamanho dos textos determinado pelo o valor especificado

Por exemplo: O dataset do CSV `dataset_100_palavras.csv` possui apenas as 100 primeiras palavras de cada texto e o dataset do CSV `dataset_200_palavras.csv` possui apenas as 200 primeiras palavras de cada texto

Os modelos de Redes Neurais Artificiais apenas trabalham com números, por isso deve ser realizado a conversão

O conversão é realizada com o auxílio da biblioteca [Gensim](https://radimrehurek.com/gensim/) utilizando o modelo Doc2Vec

Para criar os datasets, na pasta raiz do projeto digite:

`python3 script_01_convert.py`

**Resultados**

Tempo de execução: 2.55 minutos

## Detection

A detecção das falsas e verdadeiras notícias é realizado utilizando o script `script_02_detection.py`

Para criar a detecção do dataset, na pasta raiz do projeto digite:

`python3 script_02_detection.py`

Esse script também realiza a criação das imagems da pasta `graphics`

As imagems são gráficos de métricas para avaliar o desempenho da detecção

**Resultados**

Tempo de execução: 0.46 minutos
