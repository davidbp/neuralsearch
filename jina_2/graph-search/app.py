import os
import sys
from typing import Tuple
import dgl
import numpy as np
import torch

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
        self._darray_chunks = None
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
            for distances_row, idx_row in zip(dist_query_to_emb, idx): 
                for i, distance in zip(idx_row, distances_row):
                    matching_chunk = Document(self._darray_chunks[int(i)], copy=True, score=distance)
                    query_chunk.matches.append(matching_chunk)

        self._rank(docs)

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

    def _rank(self, docs, **kwargs):
        """
        Rank the queries in docs. For each query in docs get the top k matches.
        """
        for query in docs:

            parent_ids = defaultdict(list)
            for chunk in query.chunks:
                for match in chunk.matches:
                    parent_ids[match.parent_id].append(match.score.value)

            for id in parent_ids.keys():
                match = self._docs[self.docid_to_docpos[id]]
                match.score = np.min(parent_ids[id])
                query.matches.append(match)

            query.matches.sort(key=lambda x: x.score.value)

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

if __name__ == '__main__':

    if sys.argv[1] == 'index':
        print('indexing started')
        dataset = load_dataset()
        documents = create_docs(dataset)

        f = Flow().add(uses=MoleculeEncoder).add(uses=Indexer)
        with f:
            print('flow posted')
            f.post('/index',
                   inputs=documents)


    elif sys.argv[1] == 'search':

        f = Flow().add(uses=MoleculeEncoder).add(uses=Indexer)
        with f:
            for i in range(4):
                query = create_docs(data_files=[f'./data/query-{i}.mp3'])
                f.post('/search',
                       inputs=query,
                       parameters={'top_k': 10, 'distance': 'euclidean'},
                       on_done='')
    else:
        raise NotImplementedError(f'unsupported mode {sys.argv[1]}')