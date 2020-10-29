import time
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import utils

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

def dataProcessing(df):
    '''
    Método responsável por realizar o processamento dos textos convertendo o texto para o formato numérico
    '''
    # Realiza a construção das sentenças
    x = constructSentences(df['text'])

    # Monta o modelo Doc2Vec
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

    return model

def generateCSV(df, model):
    '''
    Método responsável por realizar a geração do CSV com os textos em representação numérica
    '''
    words = {}
    # Cria o novo dataframe
    df_converted = pd.DataFrame()
    # Itera em cada notícia e realiza a inserção no CSV
    for i in range(len(model.docvecs)):
        # Converte o array de palavras do texto para um dicionário

        words = { f'word_{j}' : model.docvecs.vectors_docs[i][j] for j in range(VECTOR_DIMENSION) }
        # Adicionado o dicionário junto com as outras props
        df_converted = df_converted.append(
            {'ID': df['ID'][i], 'fake_news': df['fake_news'][i], **words},
            ignore_index=True,
        )
        words = {}

    # Realiza a criação do novo CSV
    df_converted.to_csv('dataset_converted.csv')

try:
    print('Iniciando a conversão do dataset para representação numérica')
    inicio = time.time()

    # Realiza a leitura do CSV
    # df = pd.read_csv('dataset_text.csv', index_col=0)
    df = pd.read_csv('dataset_text_10_news.csv', index_col=0)

    # Realiza o conversão dos textos para representação numérica
    print('Realizando a conversão dos textos para representação numérica... ')
    model = dataProcessing(df)

    # Realiza a geração do CSV
    print('Realizando a geração do CSV... ')
    generateCSV(df, model)

    fim = time.time()
    print('CSV com o texto formatado criado com sucesso! ')
    print('Tempo de execução: %f minutos' %((fim - inicio) / 60))
except Exception as e:
    print('Falha ao gerar CSV: %s' %str(e))
