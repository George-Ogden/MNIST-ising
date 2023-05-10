import numpy as np

from src.ising import IsingModel

def test_ising_node_updates():
    model = IsingModel((2,2))
    original_nodes = model.nodes.copy()
    count = int(model.count)
    model._update(np.array([[True, False], [False, True]]))
    assert model.count == count + 1
    for i, (old_node, new_node) in enumerate(zip(original_nodes, model.nodes)):
        if i == 0 or i == 3:
            assert new_node == old_node + 1
        else:
            assert new_node == old_node

def test_ising_edge_updates():
    model = IsingModel((2,2))
    original_edges = model.edges.copy()
    count = int(model.count)
    model._update(np.array([[True, False], [False, True]]))
    assert model.count == count + 1
    for i in range(4):
        for j in range(4):
            if abs(1.5 - i) == abs(1.5 - j):
                assert model.edges[i, j] == original_edges[i, j] + 1
            else:
                assert model.edges[i, j] == original_edges[i, j]

def test_ising_fit():
    model = IsingModel((2,2))
    model.fit(np.array([[[True, False], [False, True]]]), epochs=2)
    assert model.count == 2
    assert (model.nodes == np.array([2, 0, 0, 2])).all()
    assert (model.edges == np.array([[2, 0, 0, 2], [0, 2, 2, 0], [0, 2, 2, 0], [2, 0, 0, 2]])).all()