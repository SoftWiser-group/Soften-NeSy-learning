import numpy as np
import networkx as nx
from nn_utils import *
import random
from itertools import combinations, groupby

class Graph:

    def __init__(self, n, m, adj):
        self.n = n
        self.m = m
        self.inf = 10000000000
        self.input = np.zeros((n,n))
        self.g = nx.Graph()

    def add_edges(self, a, b):
        w = np.random.randint(1, max_weight+1)
        self.g.add_edge(a, b, weight=w)
        self.g.add_edge(b, a, weight=w)
        self.input[a,b] = w
        self.input[b,a] = w

    def gnp_random_connected_graph(self):
        """
        Generates a random undirected graph, similarly to an Erdős-Rényi 
        graph, but enforcing that the resulting graph is conneted
        """
        edges = combinations(range(self.n), 2)
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        p = (float(self.m) / (self.n*self.n)) * 2.0
        if p <= 0:
            return G
        if p >= 1:
            return nx.complete_graph(n, create_using=G)
        for _, node_edges in groupby(edges, key=lambda x: x[0]):
            node_edges = list(node_edges)
            random_edge = random.choice(node_edges)
            u, v = random_edge
            # G.add_edge(*random_edge)
            w = np.random.randint(1, max_weight+1)
            G.add_edge(u, v, weight=w)
            self.input[u,v] = w
            self.input[v,u] = w
            for e in node_edges:
                if random.random() < p:
                    # G.add_edge(*e)
                    u, v = e
                    w = np.random.randint(1, max_weight+1)
                    G.add_edge(u, v, weight=w)
                    self.input[u,v] = w
                    self.input[v,u] = w
        self.g = G
    
    def construction_finished(self):
        paths = {str(i):[] for i in range(self.n)}
        for v in range(self.n):
            path = nx.all_shortest_paths(self.g, 0, v, weight='weight')
            for p in path:
                p.reverse()
                paths[str(v)].append(p)
        self.paths = paths
        self.get_dists()

    def get_dists(self):
        self.d = -np.ones((self.n, self.n)) # dists

        for i in range(self.n):
            self.d[i,i] = 0.0
        for idx, (a, b) in enumerate(self.g.edges):
            self.d[a,b] = self.g[a][b]['weight']
            self.d[b,a] = self.g[b][a]['weight']

        # compute short path distance
        for z in range(self.n):
            for x in range(self.n):
                for y in range(self.n):
                    if self.d[x,z] < 0 or self.d[z,y] < 0:
                        continue
                    if self.d[x,y] < 0:
                        self.d[x,y] = self.d[x,z] + self.d[z,y]
                    else:
                        self.d[x,y] = np.minimum(self.d[x,y], self.d[x,z] + self.d[z,y])
            
 
    @staticmethod
    def gen_random_graph(n, m):
        retg = Graph(n, m, {})
        
        # retg.gnp_random_connected_graph()

        assert m >= n - 1
        for j in range(1,n):
            i = np.random.randint(j)
            retg.add_edges(i, j)

        for i in range(m - (n-1)):
            i, j = 0, 0
            while i == j or (i, j) in retg.g.edges:
                i = np.random.randint(n)
                j = np.random.randint(n)
            retg.add_edges(i, j)

        retg.construction_finished()
        return retg
