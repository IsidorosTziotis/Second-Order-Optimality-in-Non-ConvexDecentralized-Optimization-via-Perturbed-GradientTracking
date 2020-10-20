import math
import networkx as nx

def create_graph(num_nodes):
    p = (math.log(num_nodes)) / (num_nodes - 1)
    G = nx.gnp_random_graph(num_nodes, p)
    if not nx.is_connected(G):
        print(nx.is_connected(G))
        for i in range(num_nodes - 1):
            if not G.has_edge(i, i + 1):
                G.add_edge(i, i + 1)
    print(nx.is_connected(G))
    return G
