from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import wikics
import json

def data_loader(verbose=True):

    dataset = wikics.WikiCS('./wiki-cs-dataset_autodownload')
    metadata = json.load(open('wiki-cs-dataset/dataset/metadata.json'))
    data = dataset[0] 

    return data, dataset, metadata
