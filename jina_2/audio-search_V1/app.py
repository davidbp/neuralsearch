import os
import sys
import glob

from utils import read_wav, read_mp3
from executors import VggishSegmenter, VggishEncoder, Indexer
from jina import Executor, DocumentArray, requests, Document, Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))

def create_docs( data_files):
    docs = []
    for file_path in data_files:
        if file_path.endswith('.wav'):
            data, sample_rate = read_wav(file_path)
        elif file_path.endswith('.mp3'):
            data, sample_rate = read_mp3(file_path)
        else:
            raise TypeError('create_docs expects files to be .mp3 or .wav')

        docs.append(Document(blob=data, tags={'sample_rate':sample_rate, 'file':file_path}))

    return DocumentArray(docs)

def print_indices(response):
    for doc in response.docs:
        for match in doc.matches:
            print(f'\n\tfile={match.tags["file"]}, score={match.score}')

data_folder = os.path.join(cur_dir, 'data')

def print_numchunks(request):
    print(f'\n\nlen chunks:{len(request.docs[0].chunks)}')


if __name__ == '__main__':

    if sys.argv[1] == 'index':
        documents = create_docs(data_files= glob.glob('./data/BillGatesSample-*') + glob.glob('./data/Beethoven_*.wav'))

        f = Flow().add(uses=VggishSegmenter).add(uses=VggishEncoder).add(uses=Indexer)
        with f:
            f.post('/index',
                   inputs=documents,
                   on_done=print_numchunks)

    elif sys.argv[1] == 'search':

        f = Flow().add(uses=VggishSegmenter).add(uses=VggishEncoder).add(uses=Indexer)
        with f:
            for i in range(4):
                query = create_docs(data_files=[f'./data/query-{i}.mp3'])
                f.post('/search',
                       inputs=query,
                       parameters={'top_k': 10, 'distance': 'euclidean'},
                       on_done=print_indices)
    else:
        raise NotImplementedError(f'unsupported mode {sys.argv[1]}')

