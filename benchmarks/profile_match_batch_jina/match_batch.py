
from jina import Document, DocumentArray
from jina.types.arrays.memmap import DocumentArrayMemmap 
import numpy as np


@profile
def match_func(da_1, da_2, batch_size):
    da_1.match(da_2, metric='euclidean',  limit=3, batch_size=batch_size)


@profile
def create_arrays(n_1, n_2, n_features):
    x_mat_1 = np.random.random((n_1, n_features))
    da_1 = DocumentArray([Document(embedding=x) for x in x_mat_1])

    np.random.seed(1234)
    x_mat = np.random.random((n_2, n_features))
    da_2 = DocumentArrayMemmap('./')
    for x in x_mat:
        da_2.extend([Document(embedding=x) for x in x_mat])
    
    return da_1, da_2

if __name__ == '__main__':
    n_1 = 10
    n_2 = 100_000
    n_features = 256
    batch_size = 20_000

    print('\ncreating array')
    da_1, da_2 = create_arrays(n_1, n_2, n_features)

    print('\nComputing Matches')
    match_func(da_1, da_2, batch_size=batch_size)



