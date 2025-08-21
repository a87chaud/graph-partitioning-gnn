import os                      
import networkx as nx          
import numpy as np             
from torch_geometric.utils import from_networkx 
class HyperGraphConverter:
    def __init__(self, directory: str, num_partitions: int, balance_constraint: float):
        self.directory = directory
        self.num_partitions = num_partitions
        self.balance_constraint = balance_constraint
        self.train_data = []
        self.test_data = []
        self.train_data_graphs = []
        self.test_data_graphs = []
        self.test_file_names = []
        
    def load_hypergraph_file(self, file_path):
        with open(file_path, 'r') as f:
            # first line has total number of hyperedges and vertices
            first_line = f.readline().strip()
            split = first_line.split()
            num_hyperedges, num_vertices = int(split[0]), int(split[1])
            
            hyperedges = {}
            
            for i in range(num_hyperedges):
                curr = f.readline().strip()
                vertices = [int(v) for v in curr.split()]
                # hyperedges.append(vertices)
                hyperedges[num_vertices + i + 1] = vertices
                
        return hyperedges, num_vertices
    
    def load_train_and_test_data(self):
        """
        This comes from the i cmd line args
        """
        
        all_files = sorted(os.listdir(self.directory))
        
        for counter, file_name in enumerate(all_files):
            if not file_name.endswith('.hgr'):
                  continue
            full_file_path = os.path.join(self.directory, file_name)
            if os.path.isfile(full_file_path):
                data = (self.load_hypergraph_file(full_file_path))
                if counter < 10:
                    print(f'train file: {file_name}')
                    self.train_data.append(data)
                else:
                    print(f'test file: {file_name}')
                    self.test_file_names.append(file_name)
                    self.test_data.append(data)
    
    def convert_to_graph_clique(self, data: list, train: bool):
        """
        Need to convert train data hypergraphs to graphs to run GNN on it
        Using the clique method where each hyper edge is a subgraph
        """
        for train_hyperedge, num_vertices in data:
            print("Processing file")
            G = nx.Graph()
            # Get all the vertices
            vertices = set()
            for key, val in train_hyperedge.items():
                vertices.update(val)
            
            G.add_nodes_from(list(vertices))
            
            for _, vertices in train_hyperedge.items():
                if len(vertices) > 1:
                    sub_graph = nx.complete_graph(vertices)
                    G.add_edges_from(sub_graph.edges())

            degs = np.array([G.degree[n] for n in G.nodes()], dtype=np.float32)
            for node, deg in zip(G.nodes(), degs):
                G.nodes[node]['deg'] = [deg]
            data = from_networkx(G)
            data.deg = data.deg.float()
            if train:
                self.train_data_graphs.append(data)
            else:
                self.test_data_graphs.append(data)

    # maybe try star method as well
    def convert_to_graph_star(self, data: list, train: bool):
        for hyperedges, num_vertices in data:
            G = nx.Graph()
            G.add_nodes_from(range(1, num_vertices + 1), bipartite=0)

            # Each he becomes a new node -> needs to start after all the og nodes are convered
            offset = num_vertices + 1
            for i, (he, vertices) in enumerate(hyperedges.items()):
                he_id = offset + i
                G.add_node(he_id, bipartite=1)
                for v in vertices:
                    G.add_edge(he_id, v)

            degs = np.array([G.degree[n] for n in G.nodes()], dtype=np.float32)
            for node, deg in zip(G.nodes(), degs):
                G.nodes[node]['deg'] = [deg]

            data_obj = from_networkx(G)
            data_obj.deg = data_obj.deg.float()

            if train:
                self.train_data_graphs.append(data_obj)
            else:
                self.test_data_graphs.append(data_obj)