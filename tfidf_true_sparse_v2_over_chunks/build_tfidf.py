import csv
import pickle

import sklearn
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

    data_path = "./dataset/20newgroups.csv"
    X = load_data(data_path)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf_vectorizer.fit(X)

    # store the object to disk
    pickle.dump(tfidf_vectorizer, open("./pods/tfidf_vectorizer.pickle", "wb"))
    
    # load the object later
    # tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pickle", "rb"))
