import time
import pandas as pd
from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# lista de CSVs com os tamanhos correspondentes que serão convertidos
TEXT_LENGTH = [50, 100, 150, 200]
PATH_DATASETS_FORMATTED = 'datasets/formatted/'
PATH_DATASETS_CONVERTED = 'datasets/converted/'


def constructSentences(data):
    """
    Método que realiza a construção do array de sentenças que será utilizado no doc2vec
    """
    sentences = []
    for index, row in data.iteritems():
        # converte para um formato legível para o computador
        sentences.append(TaggedDocument(utils.to_unicode(row).split(), [str(index)]))
    return sentences


def dataProcessing(df, vector_dimension):
    """
    Método responsável por realizar o processamento dos textos convertendo o texto para o formato numérico

    :param df: Dataframe com as notícias
    :type df: Dataframe
    :param vector_dimension: Quantidade de palavras que serão convertidas nos textos
    :type vector_dimension: int
    :return: Retorna o modelo Doc2Vec
    :rtype: Doc2Vec
    """
    # realiza a construção das sentenças
    x = constructSentences(df['text'])

    # monta o modelo Doc2Vec
    model = Doc2Vec(
        min_count=1,
        window=5,
        vector_size=vector_dimension,
        sample=1e-4,
        negative=5,
        workers=7,
        epochs=10,
        seed=1,
    )
    model.build_vocab(x)
    model.train(x, total_examples=model.corpus_count, epochs=model.epochs)

    return model


def generateCSV(df, model, vector_dimension):
    """
    Método responsável por realizar a geração do CSV com os textos em representação numérica

    :param df: Dataframe com as notícias
    :type df: Dataframe
    :param model: Modelo Doc2Vec
    :type model: Doc2Vec
    :param vector_dimension: Quantidade de palavras que serão convertidas nos textos
    :type vector_dimension: int
    :return: Retorna o CSV convertido
    :rtype: DataFrame
    """
    # realiza a criação das colunas do novo CSV
    columns = ['ID', 'fake_news', *[f'word_{i}' for i in range(vector_dimension)]]

    # cria o novo dataframe
    df_converted = pd.DataFrame(columns=columns)
    # itera em cada notícia e realiza a inserção no CSV
    for i in range(len(model.docvecs)):
        # converte o array de palavras do texto para um dicionário
        words = {f'word_{j}': model.docvecs.vectors_docs[i][j] for j in range(vector_dimension)}

        # adicionado o dicionário junto com as outras props
        df_converted = df_converted.append(
            {'ID': df['ID'][i], 'fake_news': df['fake_news'][i], **words},
            ignore_index=True,
        )

    # realiza as conversão de algumas colunas para valores inteiros
    df_converted[['ID', 'fake_news']] = df_converted[['ID', 'fake_news']].astype('int64')

    return df_converted


def main():
    """
    Método main do script
    """
    try:
        print('Iniciando a conversão dos datasets para representação numérica... ')
        inicio = time.time()

        # realiza a conversao dos CSVs listados em TEXT_LENGTH
        for dataset_atual in TEXT_LENGTH:
            dataset_nome = 'dataset_%i_palavras.csv' % dataset_atual
            print('Convertendo o CSV %s...' % dataset_nome)

            # realiza a leitura do CSV
            df = pd.read_csv(PATH_DATASETS_FORMATTED + dataset_nome, index_col=0)

            # realiza o conversão dos textos para representação numérica
            print('Realizando a conversão dos textos para representação numérica... ')
            model = dataProcessing(df, dataset_atual)

            # realiza a geração do CSV
            print('Realizando a geração do CSV... ')
            df_converted = generateCSV(df, model, dataset_atual)
            # realiza a criação do novo CSV
            df_converted.to_csv(PATH_DATASETS_CONVERTED + dataset_nome)

        fim = time.time()
        print('CSVs com os textos formatados criados com sucesso! ')
        print('Tempo de execução: %.2f minutos' % ((fim - inicio) / 60))
    except Exception as e:
        print('Falha ao gerar CSV: %s' % str(e))


if __name__ == '__main__':
    main()
