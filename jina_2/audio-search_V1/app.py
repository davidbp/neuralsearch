import os
import sys

from executors import VggishSegmenter, VggishEncoder, Indexer
from utils import read_wav, read_mp3

from jina import Executor, DocumentArray, requests, Document, Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))

def create_docs(data_folder, data_files):
    docs = []
    for file in data_files:
        file_path = os.path.join(data_folder, file)
        if file_path.endswith('.wav'):
            data, sample_rate = read_wav(file_path)
        elif file_path.endswith('.mp3'):
            data, sample_rate = read_mp3(file_path)
        else:
            raise TypeError('create_docs expects files to be .mp3 or .wav')

        docs.append(Document(blob=data, tags={'sample_rate':sample_rate, 'file':file}))

    return DocumentArray(docs)

def print_indices(response):
    for doc in response.docs:
        for match in doc.matches:
            print(f'\n\tfile={match.tags["file"]}, score={match.score}')

data_folder = os.path.join(cur_dir, 'data')
#documents = create_docs(data_folder, data_files=['Beethoven_1.wav', 'Beethoven_2.wav'])

def print_numchunks(request):
    print(f'\n\nlen chunks:{len(request.docs[0].chunks)}')


if __name__ == '__main__':

    if sys.argv[1] == 'index':
        documents = create_docs(data_folder, data_files=[f'BillGatesSample-{i}.mp3' for i in range(4)])

        f = Flow().add(uses=VggishSegmenter).add(uses=VggishEncoder).add(uses=Indexer)
        with f:
            f.post('/index',
                   inputs=documents,
                   on_done=print_numchunks)

    elif sys.argv[1] == 'search':
        query = create_docs(data_folder, data_files=['query.mp3'])

        f = Flow().add(uses=VggishSegmenter).add(uses=VggishEncoder).add(uses=Indexer)
        with f:
            f.post('/search',
                   inputs=query,
                   parameters={'top_k': 3, 'distance': 'cosine'},
                   on_done=print_indices)
    else:
        raise NotImplementedError(f'unsupported mode {sys.argv[1]}')

