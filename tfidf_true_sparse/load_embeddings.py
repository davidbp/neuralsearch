
import gzip
import numpy as np


def load_embeddings(path, num_dim):
    
    with gzip.open(path, 'rb') as fp:
        #import pdb;pdb.set_trace()
        b = fp.read()
        return np.frombuffer(b, dtype=np.float64).reshape([-1, num_dim])


path = "./workspace/vec.gz"
num_dim = 8597

X = load_embeddings(path, num_dim)
print(f"\n\nshape of vec.gz is {X.shape}")