import numpy as np

from src.ising import IsingModel

def test_ising_node_updates():
    model = IsingModel((2,2))
    original_nodes = model.nodes.copy()
    model._update(np.array([[True, False], [False, True]]))
    for i, (old_node, new_node) in enumerate(zip(original_nodes, model.nodes)):
        if i == 0 or i == 3:
            assert np.all(new_node == old_node + [0, 1])
        else:
            assert np.all(new_node == old_node + [1, 0])