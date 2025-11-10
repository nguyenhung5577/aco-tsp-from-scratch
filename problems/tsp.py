from typing import List
import numpy as np

def read_weight_matrix(path: str) -> np.ndarray:
    """Read a weight matrix from csv."""
    D = np.loadtxt(path, delimiter=',', dtype = float)
    assert D.shape[0] == D.shape[1], "Weight matrix must be square."
    return D

def tour_length(tour: List[int], D: np.ndarray) -> float:
    """Calculate the length of a tour"""
    length = 0.0
    n = len(tour)
    for i in range(n - 1):
        length += D[tour[i], tour[i+1]]
    return length

def random_tour(n: int, rng: np.random.Generator) -> List[int]:
    """Generate a random tour for GA, SA, HC"""
    tour = list(range(n))
    rng.shuffle(tour)
    tour.append(tour[0])
    return tour


