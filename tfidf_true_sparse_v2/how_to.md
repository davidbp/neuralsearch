# Document recommendation



## Overview

This article shows how to build a document recommendation sistem in Jina using tf-idf sparse feature vectors to represent the documents. This technique is common in use cases such as 'newspaper recommendation articles' where users that are reading a particular article have a section of  'similar recommended articles'.



## Step-by-step guide

### Step 1: Build a tf-idf vectorizer and store it to disc

We will start by "training" a class that learns the vocabulary and weights of a corpus and is able to encode text as sparse tf-idf vectors. We will use `sklearn.feature_extraction.text.TfidfVectorizer`  to do this, in this example we will be using the `20newgroups` dataset but you can change it with any corpus that you might want to use.

```python
import csv
import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(data_path):
    data_list = []
    with open(data_path) as f:
        reader = csv.reader(f, delimiter='\t')
        for i, data in enumerate(reader):
            data_list.append(data[0])   
    return data_list

if __name__ == '__main__':
    data_path = "./dataset/20newgroups.csv"
    X = load_data(data_path)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X)
    pickle.dump(tfidf_vectorizer, open("./pods/tfidf_vectorizer.pickle", "wb"))
```



### Step 2: Define your main app.py  

The core of the Jina application will be  `app.py` which will consist on a Flow defined with two `.yml`. Files  `index.yml` and `query.yml`  define the building blocks used to index and search respectively.

#### Indexing

The following  `index`  funcion makes use of the encoder defined in `index.yml` . In this case `index.yml`  defines the encoder in `pods/encode.yml` where you can see that `!TFIDFTextEncoder` is used in the first row of the file `encode.yml`.

```python
def index_generator():
    import csv
    data_path = os.path.join(os.path.dirname(__file__), os.environ['JINA_DATA_PATH'])
    with open(data_path) as f:
        reader = csv.reader(f, delimiter='\t')
        for i, data in enumerate(reader):
            d = Document()
            d.tags['id'] = int(i)
            d.text = data[0]
            yield d

def index():
    f = Flow.load_config('flows/index.yml')
    with f:
        f.index(input_fn=index_generator, batch_size=16)
```

#### Searching

The following `search` function recieves as input a path to a text file that will be used as query to find similar text documents. Here the function `print_resp` simpy shows the user the different similar documents to the input query.

```python
def print_resp(resp, document):
    for d in resp.search.docs:
        print(f"\n\n\nRanked list of related documents to the input query: \n")
        for idx, match in enumerate(d.matches):
            print('='*80)
            score = match.score.value
            if score < 0.0:
                continue
            answer = match.text.strip()
            print(f'> {idx+1:>2d}. "{answer}"\n Score: ({score:.2f})')
            print('='*80)
    
def search():
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
```



### Step 3: Index your data

So far we have only seen the main parts of the application but we have not run anything.

With the command  `python app.py index` we will run and execute the  `index` function that we have seen in Step 2. After this function is called, a folder`workspace` should be created in the folder containing `app.py`. This folder contains the embeddings and the indices needed to do search efficiently.



### Step 4: Get recommendations for a given document

With the command  `python app.py search` the application will call  the `search` function seen in Step 2 and will show the user:

```
Please type a file path for a query document:
```

Here we can type `query.txt`  and we will get the related documents to `query.txt`.




