import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#   A=  [0   a12 a13 0   a15 0  ;
#        a21 0   a23 0   0   0  ;
#        a31 a32 0   a34 0   0  ;
#        0   0   a43 0   0   0  ;
#        a51 0   0   0   0   a56;
#        0   0   0   0   a56 0  ]
edges = [[1, 2], [1, 3], [1, 5], [2, 3], [3, 4], [5, 6]]

# Create graph
g = nx.Graph()

# Add nodes
for node_index in range(6):
    g.add_node(node_index + 1)

# Add edges with weights
A = np.zeros(shape=[6, 6])
for edge in edges:
    i = edge[0] - 1
    j = edge[1] - 1
    value = np.random.random()
    A[i, j] = value
    A[j, i] = value

    g.add_weighted_edges_from(ebunch_to_add=[(edge[0], edge[1], value)])

pos = [(0, 0), (2, 1.1), (1.5, 2), (1, 1), (0, 0), (3, 1.2), (4, 2)]
nx.draw_networkx_edges(g, pos=pos, with_labels=True, edge_color='black', alpha=0.8,
                       font_size=10, width=[float(v['weight'] * 10) for (r, c, v) in g.edges(data=True)])
nx.draw_networkx_nodes(g, pos=pos, with_labels=True)
nx.draw_networkx_labels(g, pos=pos)
plt.show()
print(A)
