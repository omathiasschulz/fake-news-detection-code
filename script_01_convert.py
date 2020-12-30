import time
import pandas as pd
from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

VECTOR_DIMENSION = 300


def constructSentences(data):
    """
    Método que realiza a construção do array de sentenças que será utilizado no doc2vec
    """
    sentences = []
    for index, row in data.iteritems():
        # Converte para um formato legível para o computador
        sentences.append(TaggedDocument(utils.to_unicode(row).split(), [str(index)]))
    return sentences


def dataProcessing(df):
    """
    Método responsável por realizar o processamento dos textos convertendo o texto para o formato numérico

    :param df: Dataframe com as notícias
    :type df: Dataframe
    :return: Retorna o modelo Doc2Vec
    :rtype: Doc2Vec
    """
    # realiza a construção das sentenças
    x = constructSentences(df['text'])

    # monta o modelo Doc2Vec
    model = Doc2Vec(
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
    """
    Método responsável por realizar a geração do CSV com os textos em representação numérica

    :param df: Dataframe com as notícias
    :type df: Dataframe
    :param model: Modelo Doc2Vec
    :type model: Doc2Vec
    """
    # realiza a criação das colunas do novo CSV
    columns = ['ID', 'fake_news', *[f'word_{i}' for i in range(VECTOR_DIMENSION)]]

    # cria o novo dataframe
    df_converted = pd.DataFrame(columns = columns)
    # itera em cada notícia e realiza a inserção no CSV
    for i in range(len(model.docvecs)):
        # converte o array de palavras do texto para um dicionário
        words = {f'word_{j}': model.docvecs.vectors_docs[i][j] for j in range(VECTOR_DIMENSION)}

        # adicionado o dicionário junto com as outras props
        df_converted = df_converted.append(
            {'ID': df['ID'][i], 'fake_news': df['fake_news'][i], **words},
            ignore_index=True,
        )

    # realiza as conversão de algumas colunas para valores inteiros
    df_converted[['ID', 'fake_news']] = df_converted[['ID', 'fake_news']].astype('int64')

    # realiza a criação do novo CSV
    df_converted.to_csv('dataset_converted.csv')


def main():
    """
    Método main do script
    """
    try:
        print('Iniciando a conversão do dataset para representação numérica')
        inicio = time.time()

        # realiza a leitura do CSV
        df = pd.read_csv('dataset.csv', index_col=0)

        # realiza o conversão dos textos para representação numérica
        print('Realizando a conversão dos textos para representação numérica... ')
        model = dataProcessing(df)

        # realiza a geração do CSV
        print('Realizando a geração do CSV... ')
        generateCSV(df, model)

        fim = time.time()
        print('CSV com o texto formatado criado com sucesso! ')
        print('Tempo de execução: %f minutos' % ((fim - inicio) / 60))
    except Exception as e:
        print('Falha ao gerar CSV: %s' % str(e))


if __name__ == '__main__':
    main()
