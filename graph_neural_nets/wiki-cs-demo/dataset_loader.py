from torch_geometric.datasets import wikics
import json


def data_loader():
    dataset = wikics.WikiCS('./wiki-cs-dataset_autodownload')
    metadata = json.load(open('wiki-cs-dataset/dataset/metadata.json'))
    data = dataset[0]

    return data, dataset, metadata
