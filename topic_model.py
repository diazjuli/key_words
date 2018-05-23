import gensim
from gensim import corpora
from gensim import models
from stop_words import get_stop_words
from collections import defaultdict
from word_dictionary import word_dictionary
import sys
import os.path
import re
from itertools import chain

# this class creates different corpuses
class text_vector():
    # tokens is a list of words the text
    def tokens(text):
        # remove special characters
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

    # this creates a bag of words corpus (vector)
    def bow_corpus(dictionary, directory):
        corpus = my_corpus(dictionary, directory)
        return corpus

    #store bow corpus to bow_path
    def save_bow(corpus, bow_path):
        corpora.MmCorpus.serialize(bow_path, corpus)

    # load bow corpus from bow_path
    def load_bow(bow_path):
        corpus = corpora.MmCorpus(bow_path)
        return corpus

    # print_corpus prints the corpus_vector
    # could be a bow or tfidf vector
    def print_corpus(corpus):
        for i in corpus:
            print(i)

    # train_tfidf trains the tfidf model on a bow corpus
    def train_tfidf(bow_corpus):
        tfidf = gensim.models.TfidfModel(bow_corpus)
        return tfidf

    # bow_to_tfidf converts a bow corpus vector into a tfidf vector
    def bow_to_tfidf(tfidf_model, bow_corpus):
        tfidf_vector = tfidf_model[bow_corpus]
        return tfidf_vector

    def lsi_model(tfidf_corpus, dictionary, num_topics = 2):
        lsi = gensim.models.lsimodel.LsiModel(corpus = tfidf_corpus, id2word = dictionary, num_topics = num_topics)
        return lsi

    def tfidf_to_lsi(lsi_model, tfidf_corpus):
        lsi_vector = lsi_model[tfidf_corpus]
        return lsi_vector

    def print_lsi_model(lsi_model, num_topics = 2):
        return lsi_model.print_topics(num_topics)

    def lda_model(tfidf_corpus, dictionary, num_topics = 20, update_every = 1,chunksize = 3, passes = 1):
        lda = gensim.models.ldamodel.LdaModel(corpus = tfidf_corpus, id2word = dictionary,
                                              num_topics = num_topics, update_every = update_every,
                                              chunksize = chunksize, passes = passes)
        return lda

    def tfidf_to_lda(lda_model, tfidf_corpus):
        lda_corpus = lda_model[tfidf_corpus]
        return lda_corpus

    def print_lda_model(lda_model, num_topics = 10):
        return lda_model.print_topics(num_topics)

    def save_lda(lda, file_path):
        lda.save(file_path)

    def load_lda(lda_path):
        lda = models.LdaModel.load(lda_path)
        return lda

    def build_model(root, dict_path, bow_path, lda_path):

        word_dictionary.build_dictionary(root, dict_path)
        dictionary = word_dictionary.load_dict(dict_path)
        print(dictionary.token2id)

        if not os.path.isfile(lda_path):

            if not os.path.isfile(bow_path):
                corpus = text_vector.bow_corpus(dictionary, root)
                text_vector.save_bow(corpus, bow_path)
            else:
                corpus = text_vector.load_bow(bow_path)

            tfidf_model = text_vector.train_tfidf(corpus)
            tfidf_vector = text_vector.bow_to_tfidf(tfidf_model, corpus)

            lda = text_vector.lda_model(tfidf_vector, dictionary, num_topics = 20, update_every = 1, chunksize = 30, passes = 4)
            text_vector.save_lda(lda, lda_path)
            return lda

        else:
            lda = text_vector.load_lda(lda_path)
            return lda

    def get_topics(lda_model, num_topics):
        topics = text_vector.print_lda_model(lda_model, 10)
        return topics

    def get_doc_topic(lda_model, tfidf_model, bow_document):
        document = text_vector.bow_to_tfidf(tfidf_model, bow_document)
        lda_vector = text_vector.tfidf_to_lda(lda_model, document)
        return lda_vector

    # this function indexes a tfidf_matrix (vector) for similarity queries
    def similarities_index(tfidf_vector, num_features = 12):
        index = gensim.similarities.SparseMatrixSimilarity(tfidf_vector, num_features)
        return index

    # returns documents in tfidf_matrix in order of similarity to document_to_compare
    # document to compare is a tfidf_vector of one document in the corpus
    def get_similarities(similarities_index, document_to_compare):
        similarity = similarities_index[document_to_compare]
        return list(enumerate(similarity))


# my corpus has an interator that walks through a directory and converts
# text to vectors for each document in the directory
class my_corpus(object):
    def __init__(self, dictionary, directory):
        self.dictionary = dictionary
        self.directory = directory

    # the iterator yields the vector for each document in the directory
    def __iter__(self):
        for root, dirs, filenames in os.walk(self.directory):
            for f in filenames:
                if f.endswith(".txt"):
                    yield self.dictionary.doc2bow(text_vector.tokens(open(f).read()))



if __name__ == "__main__":
    root = "C:/Users/abize/Desktop/Abizer Jafferjee/legalX/Development/smallclaimscourtontario20032013"
    dict_path = "C:/Users/abize/Desktop/Abizer Jafferjee/legalX/Development/test/main_dict.dict"
    a = my_corpus(root, dict_path)
    for i in a:
        print(a)
    bow_path = "C:/Users/abize/Desktop/Abizer Jafferjee/legalX/Development/test/bow_corpus.mm"
    lda_path = "C:/Users/abize/Desktop/Abizer Jafferjee/legalX/Development/test/lda.model"

    # lda_model = text_vector.build_model(root, dict_path, bow_path, lda_path)
    # text_vector.get_topics(lda_model, 10)
    # text_vector.get_doc_topic(lda_model, tfid )

    print("complete")
