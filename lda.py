from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from word_dictionary import word_dictionary
from stop_words import get_stop_words
from collections import defaultdict
from psycopg2 import sql
from psql import psql
import sys
import os.path
import numpy as np
import re


class my_corpus(object):
    def __init__(self, directory):
        self.directory = directory
        self.filepaths = []

    # the iterator yields the vector for each document in the directory
    def __iter__(self):
        for root, dirs, filenames in os.walk(self.directory):
            for f in filenames:
                if f.endswith(".txt"):
                    self.filepaths.append(root+"/"+f)
                    yield open(root+"/"+f).read()

    def get_filepaths(self):
        return self.filepaths


class topic_model():

    def get_dictionary(directory, dict_path):

        if os.path.isfile(dict_path):
            return word_dictionary.load_dict(dict_path)
        else:
            return word_dictionary.build_dictionary(directory, dict_path)

    def tokenizer(text):

        text = re.sub('[^A-Za-z0-9]+', ' ', text)
        text = text.split()

        # remove stopwords
        stop_words = get_stop_words('english')
        text = [i for i in text if i not in stop_words]

        # remove words thats appear once only
        frequency = defaultdict(int)
        for i in text:
            frequency[i] += 1
        tokens = [token for token in text if frequency[token] > 1 and token.isdigit() == False]

        return tokens

    def count_vectorizer(vocabulary = None, min_df = 1,
                        max_df = 150, lowercase = True,
                        stop_words = 'english', ngram_range = (3,5)):

        vectorizer = CountVectorizer(vocabulary = vocabulary, min_df = min_df, max_df = max_df,
                                    lowercase = lowercase, stop_words = stop_words,
                                    ngram_range = ngram_range, tokenizer = topic_model.tokenizer)
        return vectorizer

    def build_corpus(directory):

        return my_corpus(directory)

    def cv_transform(vectorizer, corpus):

        return vectorizer.fit_transform(corpus)

    def inverse_transform(vectorizer, vector, index):

        return vectorizer.inverse_transform(vector[index])

    def get_feature_names(vectorizer):

        return vectorizer.get_feature_names()

    def lda(no_topics = 20, learning_method = 'online', learning_decay = 0.7,
            learning_offset = 50, random_state = 0):

        lda_model = LatentDirichletAllocation(n_components = no_topics,
                    learning_method = learning_method, learning_decay = learning_decay,
                    learning_offset = learning_offset, random_state = random_state)

        return lda_model

    def lda_transform(lda_model, vector):

        return lda_model.fit_transform(vector)

    def get_topics(lda_model, cv_vectorizer, no_topic_words):

        feature_names = topic_model.get_feature_names(cv_vectorizer)

        topic_words = {}

        for topic_idx, topic in enumerate(lda_model.components_):
            id = np.argsort(topic)[::-1][:no_topic_words]
            topic_words[topic_idx] = [feature_names[i] for i in id]

        return topic_words

    def get_lda_vector(corpus, vocabulary, cv_vectorizer, lda_model):

        cv_vector = topic_model.cv_transform(cv_vectorizer, corpus)
        lda_vector = topic_model.lda_transform(lda_model, cv_vector)

        return lda_vector

    def get_doc_topics(lda_vector):

        document_topics = {}

        for document_idx, document in enumerate(lda_vector):
            topic_idx = 0
            distribution = 0
            for i,j in enumerate(document):
                if j > distribution:
                    distribution = j
                    topic_idx = i

            document_topics[document_idx] = (topic_idx, distribution)

        return document_topics

    def get_topic_words(lda_model, cv_vectorizer):

        feature_names = topic_model.get_feature_names(cv_vectorizer)

        topic_words = {}

        for topic_idx, topic in enumerate(lda_model.components_):
            word_idx = 0
            distribution = 0
            for i, j in enumerate(topic):
                if j > distribution:
                    distribution = j
                    word_idx = i

            topic_words[topic_idx] = (feature_names[word_idx], distribution)

        return topic_words

    def get_doc_topic_words(document_topics, topic_words):

        doc_topic_words = {}

        for document, topic in document_topics.items():
            topic_idx = topic[0]
            topic_distribution = topic[1]
            words = topic_words[topic_idx][0]
            words_distribution = topic_words[topic_idx][1]

            doc_topic_words[document] = [topic_idx, words, topic_distribution, words_distribution]

        return doc_topic_words

    def main(directory, dict_path):

        vocabulary = topic_model.get_dictionary(directory, dict_path)
        cv_vectorizer = topic_model.count_vectorizer()
        lda_model = topic_model.lda()
        corpus = topic_model.build_corpus(directory)
        lda_vector = topic_model.get_lda_vector(corpus, vocabulary, cv_vectorizer, lda_model)
        topics = topic_model.get_topics(lda_model, cv_vectorizer, 10)
        document_topics = topic_model.get_doc_topics(lda_vector)
        topic_words = topic_model.get_topic_words(lda_model, cv_vectorizer)

        doc_topic_words = topic_model.get_doc_topic_words(document_topics, topic_words)

        filepaths = corpus.get_filepaths()

        for doc_idx in doc_topic_words.keys():
            doc_topic_words[doc_idx].append(filepaths[doc_idx])

        return doc_topic_words

    def write_to_db(connection, doc_topic_words):

        cursor = psql.cursor(connection)

        for doc_idx, data in doc_topic_words.items():
            document_idx = doc_idx
            topic_idx = data[0]
            tag = data[1]
            topic_strength = data[2]
            word_strength = data[3]
            result = open(data[4]).read()

            query = """insert into legal.query_result (document_idx, query, topic_strength, word_strength, result, topic_idx) values (%s, '%s', %s, %s, '%s', %s);"""\
            % (document_idx, tag, topic_strength, word_strength, result, topic_idx)
            query = psql.mogrify(cursor, query)

            psql.write_to_db(cursor, query)

        psql.commit(connection)
        psql.close_cursor(cursor)
        psql.disconnect(connection)

    def build_database(params):

        print ('Building database...')
        doc_topic_words = topic_model.main(params[directory], params[dict_path])
        connection = psql.connect(params[database_name], params[host], params[user], params[password])
        topic_model.write_to_db(connection, doc_topic_words)
        print ('build complete.')


if __name__ == "__main__":

    directory = "C:/Users/abize/Desktop/Abizer Jafferjee/legalX/Development/smallclaimscourtontario20032013"
    dict_path = "C:/Users/abize/Desktop/Abizer Jafferjee/legalX/Development/code/models/dictionary.dict"
    database_name = "'legal_data'"
    host = "'localhost'"
    user = "'postgres'"
    password = "''"

    params = {directory:directory, dict_path:dict_path, database_name:database_name, host:host, user:user, password: password}
    topic_model.build_database(params)

    print('COMPLETE.')
