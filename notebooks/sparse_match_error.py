from jina import Document, DocumentArray
import scipy.sparse as sp
import numpy as np

d1 = Document(embedding=sp.csr_matrix([[0,0,0,0,1]]))
d2 = Document(embedding=sp.csr_matrix([[1,0,0,0,0]]))
d3 = Document(embedding=sp.csr_matrix([[1,1,1,1,0]]))
d4 = Document(embedding=sp.csr_matrix([[1,2,2,1,0]]))

d1_m = Document(embedding=sp.csr_matrix([[0,0.1,0,0,0]]))
d2_m = Document(embedding=sp.csr_matrix([[1,0.1,0,0,0]]))
d3_m = Document(embedding=sp.csr_matrix([[1,1.2,1,1,0]]))
d4_m = Document(embedding=sp.csr_matrix([[1,2.2,2,1,0]]))
d5_m = Document(embedding=sp.csr_matrix([[4,5.2,2,1,0]]))

da_1 = DocumentArray([d1, d2, d3, d4])
da_2 = DocumentArray([d1_m, d2_m, d3_m, d4_m, d5_m])

print('da_1.embeddings.todense()')
print(da_1.embeddings.todense())

print('sp.vstack([d.embedding for d in da_1])')
print(np.vstack([d.embedding.todense() for d in da_1]))


da_1.match(da_2, metric='euclidean', limit=3, is_sparse=True)
query = da_1[2]
print(f'query emb = {query.embedding.todense()}')
for m in query.matches:
    print('match emb =', m.embedding.todense(), 'score =', m.scores['euclidean'].value)