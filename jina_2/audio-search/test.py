import os
import jina

import executors
from executors import VggishSegmenter, VggishEncoder, Indexer

from jina import Executor, DocumentArray, requests, Document, Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))



## think:
## 1) chunks query time (do we need to crop?)
## 2) top k
## 3) make code reading audio and sending it (instead, fill Docs with mp3)


documents = DocumentArray([Document(uri=os.path.join(cur_dir, 'data', 'Beethoven_1.wav')),
                           Document(uri=os.path.join(cur_dir, 'data', 'Beethoven_2.wav'))])

def print_numchunks(request):
    print(f'\n\nlen chunks:{len(request.docs[0].chunks)}')

f = Flow().add(uses=VggishSegmenter).add(uses=VggishEncoder).add(uses=Indexer)
with f:
    f.post('/index', inputs=documents, on_done=print_numchunks)

def print_indices(response):
    #print(f'\n\nlen(matches)={len(request.docs[0].matches)}')
    #print(f'\n\tmatches[0]={request.docs[0].matches[0]}')
    #print(f'\n\tcontent={type(request.docs[0].matches[0].content)}')
    for doc in response.docs:
        for match in doc.matches:
            print(f'\n\turi={match.uri}')

#    print(f'\n\nshape_best_match={request.docs[0].matches[0].blob.shape}')
f = Flow().add(uses=VggishSegmenter).add(uses=VggishEncoder).add(uses=Indexer)
with f:
    f.post('/search', inputs=documents, on_done=print_indices) #request_size


#def print_encoding_space(request):
#    print(f'\n\nlen vectors:{request.docs[0].chunks[0].blob.shape}')
#    print(f'\n\nlen embeddings:{request.docs[0].chunks[0].embedding.shape}')
#    print(f'\n\nlen embeddings:{request.docs[0].chunks[1].embedding.shape}')
#
#
#f = Flow().add(uses=VggishSegmenter).add(uses=VggishEncoder)
#with f:
#    f.index(inputs=documents, on_done=print_encoding_space)
#
#
#
