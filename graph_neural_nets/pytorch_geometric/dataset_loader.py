
import torch_geometric

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

def cora_loader(verbose=True):
    dataset = Planetoid(root='data/Planetoid',
                        name='Cora',
                        transform=NormalizeFeatures())
    
    data = dataset[0]  # Get the first graph object.
    
    if verbose :
        print()
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
    
        print()
        print(data)
        print('===========================================================================================================')
        
        # Gather some statistics about the graph.
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Number of training nodes: {data.train_mask.sum()}')
        print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
        print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
        print(f'Contains self-loops: {data.contains_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')

    return data, dataset

