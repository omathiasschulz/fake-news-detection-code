import time
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
import nltk
nltk.download('stopwords')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

VECTOR_DIMENSION = 300

def constructSentences(data):
    '''
    Método que realiza a construção do array de sentenças que será utilizado no gensim doc2vec
    '''
    sentences = []
    for index, row in data.iteritems():
        # Converte para um formato legível para o computador
        sentences.append(TaggedDocument(utils.to_unicode(row).split(), [str(index)]))
    return sentences

def dataProcessing(data, vector_dimension=300):
    '''
    Método responsável por realizar o processamento dos textos convertendo o texto para o formato numérico
    '''
    # Realiza a limpeza de cada registro
    for i in range(len(data)):
        data.loc[i, 'text'] = textClean(data.loc[i,'text'])

    # Realiza a construção das sentenças
    x = constructSentences(data['text'])
    y = data['label'].values

    # Modelo Doc2Vec
    model = Doc2Vec (
        min_count=1, 
        window=5, 
        vector_size=vector_dimension, 
        sample=1e-4, 
        negative=5, 
        workers=7, 
        epochs=10,
        seed=1
    )
    model.build_vocab(x)
    model.train(x, total_examples=model.corpus_count, epochs=model.iter)

    # Converte os dados numéricos para um array numpy
    x = np.zeros((len(model.docvecs), vector_dimension), dtype=float)
    for i in range(len(model.docvecs)):
        x[i] = model.docvecs[str(i)]

    return x, y

try:
    print('Iniciando a conversão do dataset para representação numérica')
    inicio = time.time()

    # Realiza a leitura do CSV
    df = pd.read_csv('dataset_text.csv', index_col=0)

    # x, y = dataProcessing(data, VECTOR_DIMENSION)



    # https://www.kaggle.com/atishadhikari/fake-news-cleaning-word2vec-lstm-99-accuracy



    # # Realiza a criação do novo CSV
    # df.to_csv('dataset_processed.csv')

    fim = time.time()
    print('CSV com o texto formatado criado com sucesso! ')
    print('Tempo de execução: %f minutos' %((fim - inicio) / 60))
except Exception as e:
    print('Falha ao gerar CSV: %s' %str(e))
