
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import pickle


def load_data(data_path):
    """
    Load the data from `data_path` and return a list of strings
    """
    with open(data_path) as f:
        data = csv.reader(f, delimiter='\t')
        X = [x[1] for x in data]
    return X

if __name__ == '__main__':

    # define a `data_path`
    data_path = "./dataset/test_answers.csv"
    X = load_data(data_path)
    
    # fit text featurizer descriptor
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X)

    # store the object to disk
    pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer.pickle", "wb"))
    
    # load the object later
    # tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pickle", "rb"))



#            with gzip.open(abspath, 'rb') as fp:
#                return np.frombuffer(fp.read(), dtype=self.dtype).reshape([-1, self.num_dim])
