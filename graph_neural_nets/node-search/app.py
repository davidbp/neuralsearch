from typing import Optional

from jina import Executor, requests, Document, DocumentArray, Flow
from jina.types.document.graph import GraphDocument

import numpy as np
import torch


class UserModel(torch.nn.Module):

    def encode(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class NodeEncoder(Executor):

    def __init__(self, model_path_state_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = UserModel()
        self.model.load_state_dict(torch.load(model_path_state_dict))

    @staticmethod
    def _get_dataset(graph):
        node_features = []
        edge_list = torch.Tensor(np.vstack(graph.adjacency.row, graph.adjacency.col))
        for node in graph.nodes:
            # do something
            node_features.append(node.blob)
        return torch.Tensor(np.stack(node_features)), edge_list

    @requests
    def encode(self, docs: Optional[DocumentArray], **kwargs):
        if not docs:
            return

        for doc in docs:
            graph = GraphDocument(doc)
            node_features, adjacency = self._get_dataset(graph)
            results = self.model.encode(node_features, adjacency)
            for node, embedding in zip(graph.nodes, results):
                # apply embedding to each node
                node.embedding = embedding


def _get_input_graph():
    return [Document()]


def _get_input_request(input_id):
    return [Document(id=input_id)]


def index():
    f = Flow().add(uses=NodeEncoder).add(uses='jinahub//SimpleIndexer', uses_with={'path': 'tmp'})
    with f:
        f.index(inputs=_get_input_graph())


def search():
    def _search(f, input_id):
        resp = f.post(on='/fill_embedding', inputs=_get_input_request(input_id), return_results=True)
        new_query = resp[0].docs[0]
        results = f.search(inputs=[new_query], return_results=True)
        matches = results[0].docs[0].matches
        return matches

    f = Flow().add(uses=NodeEncoder).add(uses='jinahub//SimpleIndexer', uses_with={'path': 'tmp'})

    with f:
        # wait for input
        print(f' Enter id to recommend from')
        input_id = input()
        matches = _search(f, input_id)
        print(f' returned nodes {len(matches)}')
        for match in matches:
            print(f' match {match}')
