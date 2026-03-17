import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class MolecularGraphDataset(Dataset):
    def __init__(self, graph_files):
        self.graph_files = graph_files

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        graph = torch.load(self.graph_files[idx])
        return graph["nodes"], graph["edges"], graph["label"]


def collate_graphs(batch):
    nodes_list, edges_list, labels = zip(*batch)
    max_nodes = max(n.size(0) for n in nodes_list)
    padded_nodes = torch.zeros(len(batch), max_nodes, nodes_list[0].size(1))
    for i, nodes in enumerate(nodes_list):
        padded_nodes[i, : nodes.size(0)] = nodes
    return padded_nodes, labels


def build_training_pipeline(graph_files, batch_size=64):
    dataset = MolecularGraphDataset(graph_files)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=collate_graphs,
        pin_memory=True,
    )
    return loader
