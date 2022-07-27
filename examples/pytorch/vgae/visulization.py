import dgl
import matplotlib.pyplot as plt
import networkx as nx
import torch
from dgl.nn.pytorch.factory import KNNGraph
kg = KNNGraph(1)
x = torch.tensor([[0,1],
[1,2]])
g = kg(x)
print(g.edges())

options = {
    'node_color': 'black',
    'node_size': 20,
    'width': 1,
}
G = dgl.to_networkx(g)
# plt.figure(figsize=[15,7])
nx.draw(G, **options)
plt.show()