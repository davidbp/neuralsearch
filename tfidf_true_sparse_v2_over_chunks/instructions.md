# Instructions to build a custom encoder and use jina to perform search



#### Build a document encoder

We will start creating a `tfidf_vectorizer_jina.py` that contains a `TFIDFTextEncoder(BaseEncoder)` class.

Note that our encoder inherits from `jina.executors.encoders.BaseEncoder` and it has two important methods:

- `post_init(self)`
  - Loads a pre-trained object to perform encoding transformations.
- `encode(self, data: 'np.ndarray')`
  - Transforms an input np.ndarray to an output representation. Here rows represent examples and columns features. 



#### Create a jina app

We can create the following terminal interface to call a jina app with two important methods:

- `index()`: Index will create use our encoder to create the embeddings used to do search.
- `search()`: Will perform the search given an input `Document` 

```python

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('choose between "index/search/dryrun" mode')
        exit(1)
    if sys.argv[1] == 'index':
        config()
        index()
    elif sys.argv[1] == 'search':
        config()
        search()
    else:
        raise NotImplementedError(f'unsupported mode {sys.argv[1]}')
```



#### index()

Here the index  creates a flow and calls `f.index` which in this case is constructed from a `yml` file in `flows/index.yml`.

```python

def index():
    """
    Index data using Index Flow.
    """
    f = Flow.load_config('flows/index.yml')

    with f:
        f.index(input_fn=index_generator, batch_size=16)
```

This index function has a custom `index_generator` that users need to build to load from disk data and generate `Document` objects.

In this case the data in the original csv has two fields per row: an `id` in the first position and the text in the second position. Hence we fill in:

- ` d.tags[`id`] = int(data[0])`

- `d.text = data[1]`

The full code would be

```python
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
            d.tags['id'] = int(data[0])
            # doc
            d.text = data[1]
            yield d
```



### search()

The `search()` function  we create has the objective of searching similar vectors to a query vector.

Here we perform the search starting with an input query that is written by the user from the terminal and then we use `f.search_lines` to search the `top_k=2` documents from within our data.

```python
def search():
    """
    Search results using Query Flow.
    """
    f = Flow.load_config('flows/query.yml')

    with f:
        while True:
            text = input("Please type a question: ")
            if not text:
                break
            def ppr(x):
                print_resp(x, text)
            f.search_lines(lines=[text, ], on_done=ppr, top_k=2)
            # f.search connects to gateway,
            # 
```

Note that a function can be passed to perform specific logic once the closest candidates are retrieved with the argument `on_done`. In our case we specify `on_done = ppr` that is a function that encapsulates `print_resp` which is defined below

```python

def print_resp(resp, question):
    """
    Print response.
    """
    for d in resp.search.docs:
        print(f"🔮 Ranked list of answers to the question: {question} \n")

        # d.matches contains the closests top_k documents in order 
        # from closer to farther from the query.
        for idx, match in enumerate(d.matches):

            score = match.score.value
            if score < 0.0:
                continue
            answer = match.text.strip()
            print(f'> {idx+1:>2d}. "{answer}"\n Score: ({score:.2f})')

```

A flow is a composition of nodes (pods). Each node is a pod and might be in a machine or in different machines. 

Any flow has:

- `flow.search` 
- `flow.index`

The number of shards = number of peas. A pea is a process or a thread. executors are instanciated in peas. Communication processes:  Head pea, Tail pea Els peas, instancien executor i escolten missatges.



