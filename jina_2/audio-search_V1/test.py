import os
import jina

import executors
from executors import VggishSegmenter, VggishEncoder, Indexer
from jina import Executor, DocumentArray, requests, Document, Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))

def create_docs(data_folder, data_files):

    docs = []
    for file in data_files:
        file_path = os.path.join(data_folder, file)
        data, sample_rate = VggishSegmenter.read_wav(file_path)
        docs.append(Document(blob=data, tags={'sample_rate':sample_rate, 'file':file}))

    return DocumentArray(docs)

data_folder = os.path.join(cur_dir, 'data')
documents = create_docs(data_folder, data_files=['Beethoven_1.wav', 'Beethoven_2.wav'])

def print_numchunks(request):
    print(f'\n\nlen chunks:{len(request.docs[0].chunks)}')

f = Flow().add(uses=VggishSegmenter).add(uses=VggishEncoder).add(uses=Indexer)
with f:
    f.post('/index', inputs=documents, on_done=print_numchunks)

def print_indices(response):
    for doc in response.docs:
        for match in doc.matches:
            print(f'\n\tfile={match.tags["file"]}')

f = Flow().add(uses=VggishSegmenter).add(uses=VggishEncoder).add(uses=Indexer)
with f:
    f.post('/search', inputs=documents,parameters={'top_k':10, 'distance':'cosine'}, on_done=print_indices) #request_size

