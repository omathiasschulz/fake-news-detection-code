import time
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from wordcloud import WordCloud

VECTOR_DIMENSION = 300

def constructSentences(data):
    '''
    Método que realiza a construção do array de sentenças que será utilizado no doc2vec
    '''
    sentences = []
    for index, row in data.iteritems():
        # Converte para um formato legível para o computador
        sentences.append(TaggedDocument(utils.to_unicode(row).split(), [str(index)]))
    return sentences

def dataProcessing(data):
    '''
    Método responsável por realizar o processamento dos textos convertendo o texto para o formato numérico
    '''
    # Realiza a construção das sentenças
    x = constructSentences(data['text'])
    y = data['fake_news'].values

    # Modelo Doc2Vec
    model = Doc2Vec (
        min_count=1, 
        window=5, 
        vector_size=VECTOR_DIMENSION, 
        sample=1e-4, 
        negative=5, 
        workers=7, 
        epochs=10,
        seed=1
    )
    model.build_vocab(x)
    model.train(x, total_examples=model.corpus_count, epochs=model.epochs)

    # Converte os dados numéricos para um array numpy
    # TODO ajustar o x
    x = np.zeros((len(model.docvecs), VECTOR_DIMENSION), dtype=float)
    for i in range(len(model.docvecs)):
        x[i] = model.docvecs[str(i)]

    return x, y

try:
    print('Iniciando a conversão do dataset para representação numérica')
    inicio = time.time()

    # Realiza a leitura do CSV
    # df = pd.read_csv('dataset_text.csv', index_col=0)
    df = pd.read_csv('dataset_text_10_news.csv', index_col=0)

    x, y = dataProcessing(df)

    print('\nx')
    print(x)

    print('\ny')
    print(y)
    
    columns = ['fake_news', 'text']
    df = pd.DataFrame(columns = columns)

    # df.append()
    # news = {**{'ID': i, 'fake_news': fake_news, 'text': text}, **metadata}

    # for news, key in x.items():
    #     print(key)
    #     # df.append(news)

    # df = df.append(dict(zip(df.columns, x)), ignore_index=True)

    print(df.head())

    # https://www.kaggle.com/atishadhikari/fake-news-cleaning-word2vec-lstm-99-accuracy

    # # Realiza a criação do novo CSV
    # df.to_csv('dataset_processed.csv')

    fim = time.time()
    print('CSV com o texto formatado criado com sucesso! ')
    print('Tempo de execução: %f minutos' %((fim - inicio) / 60))
except Exception as e:
    print('Falha ao gerar CSV: %s' %str(e))
