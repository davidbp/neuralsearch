
from jina import DocumentArray
from jina import Executor, requests 

def create_docs(dataset):
    docs = []
    for molecule_str, dgl_graph, label, mask in dataset:
        tags={'molecule_str': molecule_str}
        gdoc = GraphDocument.load_from_dgl_graph(dgl_graph)
        gdoc.tags = tags
        gdoc.blob = dgl_graph.ndata['h'].detach().numpy()
        docs.append(gdoc.tags)

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
            dgl_graph = d.to_dgl_graph()
            dgl_graph = dgl.add_self_loop(dgl_graph)
            d.embedding = model.forward(dgl_graph, feats= torch.tensor(d.blob))
    