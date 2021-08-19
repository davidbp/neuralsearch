from typing import Dict, Optional, List

from jina import Executor, requests, Document, DocumentArray, Flow
from jina.types.arrays.memmap import DocumentArrayMemmap
from jina_commons import get_logger

from jina.types.document.graph import GraphDocument
from dataset_loader import cora_loader
from model import GCN

import numpy as np
import torch

# TO REMOVE
model = GCN()
torch.save(model.state_dict(), 'model.pth')


class SimpleIndexer(Executor):
    """
    A simple indexer that stores all the Document data together,
    in a DocumentArrayMemmap object
    To be used as a unified indexer, combining both indexing and searching
    """

    def __init__(
            self,
            index_file_name: str = 'nodes',
            default_traversal_paths: Optional[List[str]] = None,
            default_top_k: int = 5,
            distance_metric: str = 'cosine',
            **kwargs,
    ):
        """
        Initializer function for the simple indexer
        :param index_file_name: The file name for the index file
        :param default_traversal_paths: The default traversal path that is used
            if no traversal path is given in the parameters of the request.
            This defaults to ['r'].
        :param default_top_k: default value for the top_k parameter
        :param distance_metric: The distance metric to be used for finding the
            most similar embeddings. The distance metrics supported are the ones supported by `DocumentArray` match method.
        """
        super().__init__(**kwargs)
        self._docs = DocumentArrayMemmap(self.workspace + f'/{index_file_name}')
        self.default_traversal_paths = default_traversal_paths or ['r']
        self.default_top_k = default_top_k
        self._distance = distance_metric
        self._use_scipy = True
        if distance_metric in ['cosine', 'euclidean', 'sqeuclidean']:
            self._use_scipy = False

        self._flush = True
        self._docs_embeddings = None
        self.logger = get_logger(self)

    @requests(on='/index')
    def index(
            self,
            docs: Optional['DocumentArray'] = None,
            parameters: Optional[Dict] = {},
            **kwargs,
    ):
        """All Documents to the DocumentArray
        :param docs: the docs to add
        :param parameters: the parameters dictionary
        """
        if not docs:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        flat_docs = docs.traverse_flat(traversal_paths)
        self._docs.extend(flat_docs)
        self._flush = True

    @requests(on='/search')
    def search(
            self,
            docs: Optional['DocumentArray'] = None,
            parameters: Optional[Dict] = {},
            **kwargs,
    ):
        """Perform a vector similarity search and retrieve the full Document match
        :param docs: the Documents to search with
        :param parameters: the parameters for the search"""
        if not docs:
            return
        if not self._docs:
            self.logger.warning(
                'no documents are indexed. searching empty docs. returning.'
            )
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        flat_docs = docs.traverse_flat(traversal_paths)
        if not flat_docs:
            return
        top_k = int(parameters.get('top_k', self.default_top_k))
        flat_docs.match(
            self._docs,
            metric=self._distance,
            use_scipy=self._use_scipy,
            limit=top_k,
        )
        self._flush = False

    @requests(on='/delete')
    def delete(self, docs: DocumentArray, parameters: Optional[Dict] = {}, **kwargs):
        """Delete entries from the index by id
        :param docs: the documents to delete
        :param parameters: parameters to the request
        """
        if docs is None:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        delete_docs_ids = docs.traverse_flat(traversal_paths).get_attributes('id')
        for idx in delete_docs_ids:
            if idx in self._docs:
                del self._docs[idx]

    @requests(on='/update')
    def update(self, docs: DocumentArray, parameters: Optional[Dict] = {}, **kwargs):
        """Update doc with the same id
        :param docs: the documents to update
        :param parameters: parameters to the request
        """
        if docs is None:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        flat_docs = docs.traverse_flat(traversal_paths)
        for doc in flat_docs:
            if doc.id is not None:
                self._docs[doc.id] = doc
            else:
                self._docs.append(doc)

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """retrieve embedding of Documents by id
        :param docs: DocumentArray to search with
        """
        if not docs:
            return
        for doc in docs:
            doc.embedding = self._docs[doc.id].embedding


class NodeEncoder(Executor):

    def __init__(self, model_path_state_dict='model.pth', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = GCN()
        self.model.load_state_dict(torch.load(model_path_state_dict))
        self.model.eval()

    @staticmethod
    def _get_dataset(nodes, adjacency):
        node_features = []
        edge_list_tensor = torch.tensor(np.vstack([adjacency.row, adjacency.col]), dtype=int)
        for node in nodes:
            # do something
            node_features.append(node.blob)
        return torch.Tensor(np.stack(node_features)), edge_list_tensor

    @requests
    def encode(self, docs: Optional[DocumentArray], **kwargs):
        if not docs:
            return

        for doc in docs:
            graph = GraphDocument(doc)
            nodes = graph.nodes
            node_features, adjacency = self._get_dataset(nodes, graph.adjacency)
            results = self.model.encode(node_features, adjacency)
            for node, embedding in zip(graph.nodes, results):
                # apply embedding to each node
                node.embedding = embedding.detach().numpy()


def _get_input_graph():
    data, dataset = cora_loader()
    gd = GraphDocument()

    edges = [(int(p[0]), int(p[1])) for p in zip(data.edge_index[0].numpy(), data.edge_index[1].numpy())]

    for i, (x, y) in enumerate(zip(data.x, data.y)):
        gd.add_node(Document(id=i, blob=x.numpy(), tags={'class': int(y)}))

    nodes = gd.nodes

    for i, o in edges:
        gd.add_edge(nodes[i], nodes[o])

    return [gd]


def _get_input_request(input_id):
    return [Document(id=input_id)]


def index():
    f = Flow().add(uses=NodeEncoder).add(uses=SimpleIndexer,
                                         uses_with={'index_file_name': 'nodes',
                                                    'default_traversal_paths': ['c']},
                                         uses_metas={'workspace': 'tmp/'})
    with f:
        f.index(inputs=_get_input_graph())


def search():
    def _search(f, input_id):
        resp = f.post(on='/fill_embedding', inputs=_get_input_request(input_id), return_results=True)
        new_query = resp[0].docs[0]
        print(f' new_query {new_query}')
        results = f.search(inputs=[new_query], return_results=True)
        matches = results[0].docs[0].matches
        return matches

    f = Flow().add(uses=SimpleIndexer,
                   uses_with={'index_file_name': 'nodes',
                              'default_traversal_paths': ['r'],
                              'default_top_k': 10,
                              'distance_metric': 'cosine'}, uses_metas={'workspace': 'tmp/'})

    with f:
        # wait for input
        print(f' Enter id to recommend from\n')
        input_id = input()
        matches = _search(f, input_id)
        print(f' returned nodes {len(matches)}')
        for match in matches:
            print(f' match {match.id} with class {match.tags["class"]}')


if __name__ == '__main__':
    index()
    search()
