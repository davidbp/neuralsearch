import os
import jina

import executors
from executors import Segmenter

from jina import Executor, DocumentArray, requests, Document, Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))

documents = DocumentArray([Document(uri=os.path.join(cur_dir, 'data', 'Beethoven_1.wav'))])

f = Flow().add(uses='segmenter.yml')
with f:
    f.index(inputs=documents, on_done=print)