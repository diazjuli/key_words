import sklearn
import numpy as np
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer

def get_paths(corpus_path):
    paths = []
    for filename in os.listdir(corpus_path):
        if filename.endswith(".txt"):
            paths.append(corpus_path + filename)
    return paths

class key_words_model():
    def __init__(self, corpus_path, file_to_extract):
        self.corpus_paths = get_paths(corpus_path)
        self.file_to_extract = file_to_extract
        self.tfidf = TfidfVectorizer(input='filename', encoding='utf-8', ngram_range=(1, 2),
                                                                stop_words='english')
        self.key_words = []

    def get_key_words(self, n):
        self.tfidf.fit(self.corpus_paths)
        doc_tfidf = self.tfidf.transform([self.file_to_extract])
        feature_array = np.array(self.tfidf.get_feature_names())
        tfidf_sorting = np.argsort(doc_tfidf.toarray()).flatten()[::-1]
        self.key_words = feature_array[tfidf_sorting][:n]
        return self.key_words

if __name__ == "__main__":

    corpus_path = "C:/Users/julia/Documents/projects/Vision_Clerk/key_words/corpus/documents/"
    file_to_extract = "C:/Users/julia/Documents/projects/Vision_Clerk/key_words/corpus/test_doc.txt"
    model = key_words_model(corpus_path, file_to_extract)
    key_words = model.get_key_words(3)
    print(key_words)

### Comments
# No need to create a bag of words for the documents. You can simply use a TFIDFVectorizer
#
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
#
# You can use sklearn to instantiate tfidf object.
# Then you can feed it a list of documents, or a list of file paths
#
# For example

#list_of_file_paths = ["path/file1.txt", "path/file2.txt"]

#tfidf = sklearn.feature_extraction.text.TfidfVectorizer(input = 'filename', encoding = 'utf-8', ngram_range = (1,2), stop_words = 'english')
# the tfidf has many parameters which can be changed, just check the link. You can add a preprocessing funciton, tokenizing function, dictionary (if none is given, it creates one automatically), normalizaiton etc

# then call
#tfidf.fit(list_of_file_paths)

# all the methods you can use on the tfidf are further down in the link
# you can also see the vocabulary you used by tfidf.vocabulary_

# once you have your tfidf you can convert a document into document-term matrix by
#doc_tfidf = tfidf.transform(raw_document)

# then you can do something like this to get top 3 words based on tfidf (https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score)

#feature_array = np.array(doc_tfidf.get_feature_names())
#tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

#n = 3
#top_n = feature_array[tfidf_sorting][:n]
#print(top_n)

# there also examples of use of tfidf all the way down
