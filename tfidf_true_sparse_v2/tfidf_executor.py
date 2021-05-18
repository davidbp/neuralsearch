__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import numpy as np
import scipy
from collections import OrderedDict
import sklearn
from sklearn import datasets
from typing import Dict

from jina import Executor, requests, DocumentArray, Document
from jina import Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))



class TFIDFTextEncoder(Executor):

    def __init__(
        self,
        path_vectorizer: str = os.path.join(cur_dir, 'model/tfidf_vectorizer.pickle'),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.path_vectorizer = path_vectorizer

        import os
        import pickle

        if os.path.exists(self.path_vectorizer):
            self.tfidf_vectorizer = pickle.load(open(self.path_vectorizer, 'rb'))
        else:
            raise PretrainedModelFileDoesNotExist(
                f'{self.path_vectorizer} not found, cannot find a fitted tfidf_vectorizer'
            )

    @requests(on=['/index','/search'])
    def encode(self,docs: DocumentArray,  *args, **kwargs) -> DocumentArray:
        iterable_of_texts = docs.get_attributes('text')
        embedding_matrix = self.tfidf_vectorizer.transform(iterable_of_texts)

        for doc, doc_embedding in zip(docs, embedding_matrix):
            doc.embedding = doc_embedding


class SparseIndexer(Executor):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.id_to_embedding = OrderedDict()
        self.id_to_document = OrderedDict()

    @requests(on='/index')
    def add(self,docs:DocumentArray, *args, **kwargs):

        for doc in docs:
            self.id_to_embedding[doc.id] = doc.embedding
            self.id_to_document[doc.id] = doc

    @requests(on='/search')
    def query(self, docs: DocumentArray, parameters: Dict, *args, **kwargs):
        top_k = int(parameters['top_k'])

        def matrix_vector_distance(X,x):
             distances = []
             n_rows = X.shape[0]
         
             for k in range(n_rows):
                 aux = X[k] - x
                 distances.append((aux).multiply(aux).mean())
                 
             return distances

        X_embeddings = scipy.sparse.vstack([scipy.sparse.csr_matrix(emb) for emb in self.id_to_embedding.values()])
        indices = list(self.id_to_document.keys())
            
        for query in docs:
            # get closest top_k vector
            query_emb = scipy.sparse.csr_matrix(query.embedding)
            sorted_indices = np.argsort(matrix_vector_distance(X_embeddings, query_emb))# correct order?

            # get ids of closest vector
            top_k_indices = sorted_indices[0:top_k]
            
            for index in top_k_indices:
                query.matches.append(self.id_to_document[indices[index]])

        return None

def print_embeddings(response):
    print(f'\nprint_embeddings\n\nresponse={response}\n\n\n')
    for doc in response.data.docs:
        print(doc.embedding)

def print_matches(response):
    print(f'\nprint_matches\n\nresponse={response}\n\n\n')
    for query in response.data.docs:
        print(f'\n\nquery.text={query.text}')
        for i, match in enumerate(query.matches):
            print(f'\n\n\nTOP={i}, match.text={match.text}')

def get_20newsgroup_data():
    data = sklearn.datasets.fetch_20newsgroups()
    texts = data['data']
    for text in texts:
        d = Document(text=text)
        yield d 

# Get data from sklearn
data = sklearn.datasets.fetch_20newsgroups()
texts = data['data']
x_query = DocumentArray([Document(text=texts[100])])


f = Flow().add(uses=TFIDFTextEncoder).add(uses=SparseIndexer)
with f:
   f.post(on='/index', 
          request_size=5000, 
          inputs=get_20newsgroup_data)

   f.post(on='/search', 
          request_size=640, 
          parameters={'top_k' : 4},
          on_done=print_matches,
          inputs=x_query)
