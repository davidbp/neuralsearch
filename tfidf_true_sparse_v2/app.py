__version__ = '0.0.1'

import os
import sys
from jina.flow import Flow
from jina import Document

def config():
    """
    Configure environment variables.
    """
    parallel = 1 if sys.argv[1] == 'index' else 1
    shards = 1
    os.environ['JINA_PARALLEL'] = str(parallel)
    os.environ['JINA_SHARDS'] = str(shards)
    os.environ['WORKDIR'] = './workspace'
    os.makedirs(os.environ['WORKDIR'], exist_ok=True)
    os.environ['JINA_PORT'] = os.environ.get('JINA_PORT', str(65481))
    os.environ['JINA_DATA_PATH'] = 'dataset/20newgroups.csv'

def index_generator():
    """
    Define data as Document to be indexed.
    """
    import csv
    data_path = os.path.join(os.path.dirname(__file__), os.environ['JINA_DATA_PATH'])

    # Get Document and ID
    with open(data_path) as f:
        reader = csv.reader(f, delimiter='\t')
        for i, data in enumerate(reader):
            d = Document()
            # docid
            d.tags['id'] = int(i)
            # doc
            d.text = data[0]
            yield d

def index():
    """
    Index data using Index Flow.
    """
    f = Flow.load_config('flows/index.yml')

    with f:
        #f.block()
        f.index(input_fn=index_generator, request_size=32)#batch_size=16)


def print_resp(resp, document):
    """
    Print response.
    """
    for d in resp.search.docs:
        print(f"\n\n\nRanked list of related documents to the input query: \n")

        # d.matches contains the closests top_k documents in order 
        # from closer to farther from the query.
        for idx, match in enumerate(d.matches):
            print('='*80)
            score = match.score.value
            answer = match.text.strip()
            print(f'> {idx+1:>2d}. "{answer}"\n Score: ({score:.2f})')
            print('='*80)
    
def search():
    """
    Search results using Query Flow.
    """
    f = Flow.load_config('flows/query.yml')

    with f:
        while True:
            file_path = input('Please type a file path for a query document: ')

            while os.path.exists(file_path)==False:                
                print(f'Previous file_path={file_path} cannot be found.' )
                file_path = input('Please type a file path of a document:')
            with open(file_path,'r') as file_:
                text = file_.read()
            
            def ppr(x):
                print_resp(x, text)

            f.search_lines(lines=[text, ], on_done=ppr, top_k=3, line_format=None)



if __name__ == '__main__':
    
    from tfidf_executor import TFIDFTextEncoder
    if sys.argv[1] == 'index':
        config()
        index()
    elif sys.argv[1] == 'search':
        config()
        search()
    else:
        raise NotImplementedError(f'unsupported mode {sys.argv[1]}')
