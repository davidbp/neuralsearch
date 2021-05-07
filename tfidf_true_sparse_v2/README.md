<p align="center">



#tf-idf example in jina



This example showcases how to encode text data with a tf-idf feature descriptor using sparse scipy arrays and how to do search with the sparse data.



Steps:

- run `python build_tfidf.py`
    - This script will generate `tfidf_vectorizer.pickle` which will be used to generate embeddings
- run `python app.py index`
- run `python app.py search`



