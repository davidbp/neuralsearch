import numpy as np

def _get_ones(x, y):
    return np.ones((x, y))

def _ext_A(A):
    nA, dim = A.shape
    A_ext = _get_ones(nA, dim * 3)
    A_ext[:, dim: 2 * dim] = A
    A_ext[:, 2 * dim:] = A ** 2
    return A_ext

def _ext_B(B):
    nB, dim = B.shape
    B_ext = _get_ones(dim * 3, nB)
    B_ext[:dim] = (B ** 2).T
    B_ext[dim: 2 * dim] = -2.0 * B.T
    del B
    return B_ext

def _euclidean(A_ext, B_ext):
    sqdist = A_ext.dot(B_ext).clip(min=0)
    return np.sqrt(sqdist)

def euclidean(_query_vectors, raw_B):
    data = _ext_B(raw_B)
    return _euclidean(_query_vectors, data)

def euclidean_vectorized(query_vectors, raw_B):
    _query_vectors = _ext_A(query_vectors)
    return euclidean(_query_vectors, raw_B)

def _norm(A):
    return A / np.linalg.norm(A, ord=2, axis=1, keepdims=True)

def _cosine(A_norm_ext, B_norm_ext):
    return A_norm_ext.dot(B_norm_ext).clip(min=0) / 2

def cosine(_query_vectors, raw_B):
    data = _ext_B(raw_B)
    return _cosine(_query_vectors, data)

def cosine_vectorized(query_vectors, raw_B):
    _query_vectors = _ext_A(query_vectors)
    return cosine(_query_vectors, raw_B)
