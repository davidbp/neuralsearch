import os
import sys
from typing import Tuple
import dgl
import numpy as np
import torch
import pickle

from jina.types.document.graph import GraphDocument
from jina import Executor, requests, DocumentArray, Flow
cur_dir = os.path.dirname(os.path.abspath(__file__))


def create_docs(dataset):
    docs = []
    for molecule_str, dgl_graph, label, mask in dataset:
        tags={'molecule_str': molecule_str,
              'agg_features': dgl_graph.ndata['h'].detach().numpy().tolist(),
              'label':label.detach().numpy().tolist(),
              'mask':mask.detach().numpy().tolist()}
        gdoc = GraphDocument.load_from_dgl_graph(dgl_graph)
        gdoc.tags = tags
        docs.append(gdoc)
        
    return DocumentArray(docs)


class MoleculeEncoder(Executor):
    
    def __init__(self, model_type: str='GCN_Tox21', *args, **kwargs):

        super().__init__(*args, **kwargs)
        import torch
        from dgllife.model import load_pretrained
        from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
        self.model = load_pretrained(model_type) 
        self.model.eval()

    @requests()
    def encode(self, docs: DocumentArray, *args, **kwargs):
        for d in docs:
            dgraph = GraphDocument(d)
            dgl_graph = dgraph.to_dgl_graph()
            dgl_graph = dgl.add_self_loop(dgl_graph)
            torch_features = torch.tensor(d.tags['agg_features'])
            d.embedding = self.model.forward(dgl_graph, feats=torch_features).detach().numpy()


class Indexer(Executor):

    def __init__(self, index_folder=f'{cur_dir}/workspace/', *args, **kwargs):
        self.index_folder = index_folder
        self.index_path = os.path.join(self.index_folder,'docs.json')
        self._embedding_matrix = None
        self.docid_to_docpos = None

        if os.path.exists(self.index_path):
            self._docs = DocumentArray.load(self.index_path)
            self._embedding_matrix = np.stack(self._docs.get_attributes('embedding')) 
            self.docid_to_docpos = {doc.id: i for i, doc in enumerate(self._docs)}

        else:
            self._docs = DocumentArray()

    @requests(on='/index')
    def index(self, docs: DocumentArray, *args, **kwargs):
        self._docs.extend(docs)

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters, **kwargs):
        top_k = int(parameters['top_k'])
        distance = parameters['distance']

        for query in docs:
            q_emb = np.stack(query.chunks.get_attributes('embedding'))  # get all embedding from query docs

            if distance == 'cosine':
                dist_query_to_emb = cosine_vectorized(q_emb, self._embedding_matrix)
            if distance == 'euclidean':
                dist_query_to_emb = np.linalg.norm(q_emb[:, None, :] - self._embedding_matrix[None, :, :], axis=-1)

            idx, dist_query_to_emb = self._get_sorted_top_k(dist_query_to_emb, top_k)

            # soted_idices[0] < soted_idices[1] < ...
            sorted_indices = np.argosrt(dist_query_to_emb)

            for idx in sorted_indices:
                match = Document(self._docs[self.docid_to_docpos[idx]]) 
                query.matches.append(match,copy=True, score=dist_query_to_emb[idx])

        #self._rank(docs)

    @staticmethod
    def _get_sorted_top_k(dist: 'np.array', top_k: int) -> Tuple['np.ndarray', 'np.ndarray']:
        """Find top-k smallest distances in ascending order.

        Idea is to use partial sort to retrieve top-k smallest distances unsorted and then sort these
        in ascending order. Equivalent to full sort but faster for n >> k. If k >= n revert to full sort.
        """
        if top_k >= dist.shape[1]:
            idx = dist.argsort(axis=1)[:, :top_k]
            dist = np.take_along_axis(dist, idx, axis=1)
        else:
            idx_ps = dist.argpartition(kth=top_k, axis=1)[:, :top_k]
            dist = np.take_along_axis(dist, idx_ps, axis=1)
            idx_fs = dist.argsort(axis=1)
            idx = np.take_along_axis(idx_ps, idx_fs, axis=1)
            dist = np.take_along_axis(dist, idx_fs, axis=1)

        return idx, dist

    def close(self):
        os.makedirs(self.index_folder, exist_ok = True)
        self._docs.save(self.index_path)

def load_dataset():
    from dgllife.model import GCNPredictor
    from dgllife.data import Tox21
    from dgllife.model import load_pretrained
    from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

    dataset = Tox21(smiles_to_bigraph, CanonicalAtomFeaturizer())
    return dataset


def print_indices(response):
    for doc in response.docs:
        print(f"\n\nmolecule_str={doc.tags['molecule_str']}, score={doc.score}")


if __name__ == '__main__':
    n_queries = 3

    if sys.argv[1] == 'index':
        print('indexing started')
        dataset = load_dataset()
        documents = create_docs(dataset)

        for i in range(n_queries):
            file = open(f'query_{i}.pkl','wb')
            pickle.dump(documents[i], file)
            file.close()

        f = Flow().add(uses=MoleculeEncoder).add(uses=Indexer)
        with f:
            print('flow posted')
            f.post('/index',
                   inputs=documents)

    elif sys.argv[1] == 'search':

        queries = []
        for i in range(n_queries):
            query = pickle.load(open('query_{i}','rb'))
            queries.append(query)

        f = Flow().add(uses=MoleculeEncoder).add(uses=Indexer)
        with f:
            for query in queries:
                f.post('/search',
                       inputs=query,
                       parameters={'top_k': 4, 'distance': 'euclidean'},
                       on_done=print_indices)
    else:
        raise NotImplementedError(f'unsupported mode {sys.argv[1]}')