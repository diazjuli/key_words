import gensim
from gensim import corpora
from gensim import models
from stop_words import get_stop_words
from collections import defaultdict
import sys
import os.path
import re
from itertools import chain

# this class has functions to build a word dictionary from a set of text files
class word_dictionary():
    # corpus is a list of tokens for the document
    def create_corpus(text):
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
        tokens = [token for token in text if frequency[token] > 1]
        return tokens

    # create a new dictionary from a corpus of tokens
    def create_dictionary(text):
        corpus = [word_dictionary.create_corpus(text)]
        dictionary = corpora.Dictionary(corpus)
        return dictionary

    # add new document to saved dictionary
    def add_to_dictionary(dictionary, text):
        corpus = [word_dictionary.create_corpus(text)]
        dictionary.add_documents(corpus)
        return dictionary

    # save a dictionary
    def save_dict(dictionary, path):
        dictionary.save(path)

    # load a saved dictionary
    def load_dict(path):
        dictionary = corpora.Dictionary.load(path)
        return dictionary

    # build the dictionary and dave it to dict_path file
    def build_dictionary(rootdir, dict_path = None):
        # walk through the directory
        for root, dirs, filenames in os.walk(rootdir):
            for f in filenames:
                if f.endswith(".txt"):
                    text = open(root+"/"+f).read()
                    # if the dictionary does not exist
                    if not os.path.isfile(dict_path):
                        # create a new dictionary and save it
                        dictionary = word_dictionary.create_dictionary(text)
                        word_dictionary.save_dict(dictionary, dict_path)
                        return dictionary
                    # if a dictionary exists
                    if os.path.isfile(dict_path):
                        # load the dictionary, add the new words to the dictionary
                        # and save it
                        dictionary = word_dictionary.load_dict(dict_path)
                        dictionary = word_dictionary.add_to_dictionary(dictionary, text)
                        word_dictionary.save_dict(dictionary, dict_path)
                        return dictionary
