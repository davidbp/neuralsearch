import csv
import pickle
import os

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(data_path):

    import csv
    data_list = []
    with open(data_path) as f:
        reader = csv.reader(f, delimiter='\t')
        for i, data in enumerate(reader):
            data_list.append(data[0])   
    return data_list

if __name__ == '__main__':

    dataset = datasets.fetch_20newsgroups()
    list_strings = dataset.data

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf_vectorizer.fit(list_strings)

    # store the object to disk
    os.mkdir('model')
    pickle.dump(tfidf_vectorizer, open("./model/tfidf_vectorizer.pickle", "wb"))
    