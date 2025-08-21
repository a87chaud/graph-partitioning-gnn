import argparse
import os
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
import random
import numpy as np
from torch_geometric.loader import DataLoader
from hypergraph_converter import HyperGraphConverter
from gnn import GNN

def gap_loss(Y, edge_index, num_parts):
    """
    Calculate the loss using the gap loss function from the paper
    
    As seen in fig1 there are 3 steps
    1. gamma = y^t * deg
    2. e_ncut = sum((y * gamma) * (1 - y))
    3. balance = sum(sum(y) - n/g) ^ 2
    """
    deg = torch.bincount(edge_index[0], minlength=Y.size(0)).float().unsqueeze(1)
    gamma = torch.matmul(Y.T, deg) 
    Y_norm = Y / gamma.T
    start, end = edge_index
    e_ncut = ((Y_norm[start] * (1 - Y[end])).sum(dim=1)).sum()
    balance = ((Y.sum(dim=0) - (Y.size(0) / num_parts)) ** 2).sum()

    return e_ncut + balance

def train(model, optimizer, num_partitions, loader):
    total_loss = 0
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        Y = model(batch.deg, batch.edge_index)
        loss = gap_loss(Y, batch.edge_index, num_partitions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def infer(model, data):
    model.eval()
    with torch.no_grad():
        Y = model(data.deg, data.edge_index)
        parts = torch.argmax(Y, dim=1)
    return parts.cpu().numpy()

def test(model, loader):
    model.eval()
    all_parts = []
    with torch.no_grad():
        for batch in loader:
            Y = model(batch.deg, batch.edge_index)
            curr_part = torch.argmax(Y, dim=1).cpu().numpy()
            all_parts.append(curr_part)
    return all_parts

def parse_args():
    parser = argparse.ArgumentParser(description="GNN")

    parser.add_argument(
        '-i', '--i',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-k', '--k',
        type=int,
        required=True,
        choices=range(2, 17),
    )
    parser.add_argument(
        '-p', '--p',
        type=float,
        required=True,
    )
    parser.add_argument(
        '-w', '--w',
        required=False,
    )

    args = parser.parse_args()

    if not (0 <= args.p <= 1):
        parser.error("Balance constraint (-p) must be between 0 and 1.")
    return args

def compute_hyperedge_stats(hgr_file, part_file):
    with open(hgr_file, "r") as f:
        num_hyperedges, num_vertices = map(int, f.readline().split())
        hyperedges = []
        for _ in range(num_hyperedges):
            vertices = list(map(int, f.readline().split()))
            hyperedges.append(vertices)

    with open(part_file, "r") as f:
        partitions = [int(line.strip()) for line in f]

    hyperedge_cut = 0
    sum_external_deg = 0

    for hedge in hyperedges:
        parts_in_edge = {partitions[v - 1] for v in hedge}
        if len(parts_in_edge) > 1:
            hyperedge_cut += 1
            sum_external_deg += len(parts_in_edge)
    
    print(f"Hyperedge Cut: {hyperedge_cut}")
    print(f"Sum of External Degrees: {sum_external_deg}")

def main():
    args = parse_args()
    
    hgr_dir_path = args.i
    num_partitions = args.k
    balance_constraint = args.p
    weight_file = f"weights_k{num_partitions}.bin"

    # Load data
    graph_converter = HyperGraphConverter(
        directory=hgr_dir_path,
        num_partitions=num_partitions,
        balance_constraint=balance_constraint
    )
    print("Started")
    graph_converter.load_train_and_test_data()
    print("Loaded data")
    graph_converter.convert_to_graph_star(graph_converter.train_data, True)
    graph_converter.convert_to_graph_star(graph_converter.test_data, False)
    model = GNN(in_dim=1, hidden_dim=32, num_partitions=num_partitions)
    print("Converted")
    print(f'Train data: {graph_converter.train_data_graphs[0]}')
    if args.w and os.path.exists(args.w):
        # Load weights and run infer
        model.load_state_dict(torch.load(args.w))
        print(f"Loading weights from {args.w}")
    else:
        train_loader = DataLoader(graph_converter.train_data_graphs, batch_size=1, shuffle=True)
        # test_loader = DataLoader(graph_converter.test_data_graphs, batch_size=1, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # Train
        # Reducing epochs from 100 to 50?
        for epoch in range(1, 51):
            loss = train(model, optimizer, num_partitions, train_loader)
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

        torch.save(model.state_dict(), weight_file)
        print(f"Saved weights to {weight_file}")
    hmetis_values_k2 = {
        "ibm11.hgr.part.2": (782, 1564),
        "ibm12.hgr.part.2": (1866, 3732),
        "ibm13.hgr.part.2": (729, 1458),
        "ibm14.hgr.part.2": (1708, 3416),
        "ibm15.hgr.part.2": (2164, 4328),
        "ibm16.hgr.part.2": (1675, 3350),
        "ibm17.hgr.part.2": (2226, 4452),
        "ibm18.hgr.part.2": (1567, 3134),

    }
    # Inference using the test set
    for (graph_data, file_name) in zip(graph_converter.test_data_graphs, graph_converter.test_file_names):
        parts = infer(model, graph_data)
        print("Inference done")
        part_filename = f"{file_name}.part.{num_partitions}"
        with open(part_filename, "w") as f:
            for p in parts:
                f.write(f"{p}\n")
        print(" ------------------- MY OUTPUT -------------------")
        print(f"Part file: {part_filename}")
        compute_hyperedge_stats(f'{hgr_dir_path}/{file_name}', part_filename)
        print(" #################### HMETIS OUTPUT ******************")
        he, ed = hmetis_values_k2[part_filename]
        print(f"HMETIS hyperedge cuts: {he}")
        print(f"HMETIS Sum of External Degrees: {ed}")

    
    
if __name__ == '__main__':
    main()
