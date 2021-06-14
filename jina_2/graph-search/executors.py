import numpy as np
import dgl
import torch
import os
from typing import Tuple

from jina.types.document.graph import GraphDocument
from jina import DocumentArray, Document, Executor, requests
from utils import cosine_vectorized, euclidean_vectorized

cur_dir = os.path.dirname(os.path.abspath(__file__))

class MoleculeEncoder(Executor):
    
    def __init__(self, model_type: str='GCN_Tox21', *args, **kwargs):

        super().__init__(*args, **kwargs)
        from dgllife.model import load_pretrained
        self.model = load_pretrained(model_type)
        self.model.eval()

    @requests()
    def encode(self, docs: DocumentArray, *args, **kwargs):
        for d in docs:
            dgraph = GraphDocument(d)
            dgl_graph = dgraph.to_dgl_graph()
            dgl_graph = dgl.add_self_loop(dgl_graph)
            #print(f'\n\n\n,d.tags.keys()={d.tags.keys()}\n\n\n')
            torch_features = torch.tensor(d.tags['agg_features'])
            d.embedding = self.model.forward(dgl_graph, feats=torch_features).detach().numpy().flatten()


class Indexer(Executor):

    def __init__(self, index_folder=f'{cur_dir}/workspace/', *args, **kwargs):
        self.index_folder = index_folder
        self.index_path = os.path.join(self.index_folder, 'docs.json')
        self._embedding_matrix = None
        self.docid_to_docpos = None
        self.docpos_to_docid = None

        if os.path.exists(self.index_path):
            self._docs = DocumentArray.load(self.index_path)
            self._embedding_matrix = np.stack(self._docs.get_attributes('embedding')) 
            self.docid_to_docpos = {doc.id: i for i, doc in enumerate(self._docs)}
            self.docpos_to_docid = {v: k for k, v in self.docid_to_docpos.items()}

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
            #q_emb = query.embedding  # row vector (1, n_embedding)
            q_emb = np.stack([query.get_attributes('embedding')])

            print(f'\n\n\nq_emb.shape={q_emb.shape}\n\n\n')
            print(f'\n\n\nself._embedding_matrix.shape={self._embedding_matrix.shape}\n\n\n')

            if distance == 'cosine':
                dist_query_to_emb = cosine_vectorized(q_emb, self._embedding_matrix)
            if distance == 'euclidean':
                #dist_query_to_emb = np.linalg.norm(q_emb[:, None, :] - self._embedding_matrix[None, :, :], axis=-1)
                dist_query_to_emb = euclidean_vectorized(q_emb, self._embedding_matrix)

            print(f'\n\n\ndist_query_to_emb.shape={dist_query_to_emb.shape}\n\n\n')

            idx, dist_query_to_emb = self._get_sorted_top_k(dist_query_to_emb, top_k)

            dist_query_to_emb = dist_query_to_emb.flatten()

            # soted_idices[0] < soted_idices[1] < ...
            sorted_indices = np.argsort(dist_query_to_emb)
            sorted_distances = dist_query_to_emb[sorted_indices]

            print(f'\n\n\nidx={idx}\n\n\n')
            print(f'\n\n\ndist_query_to_emb={dist_query_to_emb}\n\n\n')
            print(f'\n\n\nlen(self._docs)={len(self._docs)}\n\n\n')
            print(f'\n\n\nsorted_indices={sorted_indices}\n\n\n')
            print(f'\n\n\nsorted_distances={sorted_distances}\n\n\n')

            for id, dist in zip(sorted_indices, sorted_distances):
                #print(f'\n\n\nidx={idx}\n\n\n')
                #print(f'\n\n\ntype(self.docid_to_docpos)={self.docid_to_docpos}\n\n\n')
                #print(f'\n\n\nself.docid_to_docpos[idx]={self.docid_to_docpos[idx]}\n\n\n')
                #print(f'\n\n\ndist_query_to_emb[int(id)]={dist_query_to_emb[int(id)]}\n\n\n')
                #print(f'\n\n\nid={id}\n\n\n')
                match = Document(self._docs[int(id)], score=dist)
                print(f'\n\n\nmatch.score={match.score}\n\n\n')
                query.matches.append(match)

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

