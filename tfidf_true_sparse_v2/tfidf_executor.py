
__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import numpy as np
import scipy
from collections import OrderedDict

from jina import Executor, requests, DocumentArray, Document
#from jina.executors.encoders import BaseEncoder

cur_dir = os.path.dirname(os.path.abspath(__file__))

#from jina import Executor, requests, Flow, Document


class TFIDFTextEncoder(Executor):
    """Encode ``Document`` content from a `np.ndarray` (of strings) of length `BatchSize` into
    a `csr_matrix` of shape `Batchsize x EmbeddingDimension`.

    :param path_vectorizer: path containing the fitted tfidf encoder object
    :param args: not used
    :param kwargs: not used
    """

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
        """Encode the ``Document`` content creating a tf-idf feature vector of the input.

        :param content: numpy array of strings containing the text data to be encoded
        :param args: not used
        :param kwargs: not used
        """
        #print(f'\n\nargs={args}')
        #print(f'\n\nkwargs={kwargs}')
        #print(f'\n\ndocs={docs}')
        #print('\n\n')
        #print(f'id(docs)={id(docs)}')
        iterable_of_texts = docs.get_attributes('text')

        #docarray.embedding = self.tfidf_vectorizer.transform(iterable_of_texts)
        embedding_matrix = self.tfidf_vectorizer.transform(iterable_of_texts)

        for doc, doc_embedding in zip(docs, embedding_matrix):
            doc.embedding = doc_embedding

        # if return docs # then data is empty
        #return docs
#        return None


class SparseIndexer(Executor):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Dict[str, sparsearray]
        self.id_to_embedding = OrderedDict()

        # Dict[str, Document]
        self.id_to_document = OrderedDict()


    @requests(on='/index')
    def add(self,docs:DocumentArray, *args, **kwargs):

        for doc in docs:
            self.id_to_embedding[doc.id] = doc.embedding
            self.id_to_document[doc.id] = doc

    @requests(on='/search')
    def query(self,docs:DocumentArray, *args, **kwargs):
        top_k = 3

        print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')
        print([scipy.sparse.csr_matrix(emb) for emb in self.id_to_embedding.values()])
        print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')

        X_embeddings = scipy.sparse.vstack([scipy.sparse.csr_matrix(emb) for emb in self.id_to_embedding.values()])

        for x in X_embeddings:
            
        for query in docs:
            # get closest top_k vector
            print(X_embeddings.shape,scipy.sparse.csr_matrix(query.embedding).shape )
            sorted_indices = np.argsort(X_embeddings - scipy.sparse.csr_matrix(query.embedding))# correct order?

            # get ids of closest vector
            top_k_indices = sorted_indices[top_k]

            for index in top_k_indices:
                query.matches.add(self.id_to_document[index])

        return None




def print_embeddings(response):
    print(f'\nprint_embeddings\n\nresponse={response}\n\n\n')
    for doc in response.data.docs:
        print(doc.embedding)

def print_matches(response):
    print(f'\nprint_matches\n\nresponse={response}\n\n\n')
    for query in response.data.docs:
        print('query.text={query.text}')
        for match in query.matches:
            print('match.text={match.text}')


from jina import Flow

f = Flow().add(uses=TFIDFTextEncoder).add(uses=SparseIndexer)

texts = ['Hello Bo', 'Hello Joan', 'Hello Kelton']
docarray = DocumentArray([Document(text=x) for x in texts])


with f:
   f.post(on='/index', 
          request_size=32, 
          on_done=print_embeddings,
          inputs=docarray)

   f.post(on='/search', 
          request_size=32, 
          on_done=print_matches,
          inputs=docarray)

#f.post(on='/search', 
#          request_size=32, 
#          on_done=print_embeddings,
#          inputs=docarray)

#with f:
#   f.post(on='/search', 
#          request_size=32, 
#          on_done=print_embeddings,
#          inputs=docarray)







