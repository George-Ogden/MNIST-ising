import numpy as np

def pytest_configure(config):
    # seed for repeatability
    np.random.seed(0)