import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from problems.tsp import tour_length

Tour = List[int]
History = List[float]
AlgorithmResult = Tuple[Tour, float, History]
IterationCallback = Optional[Callable[[int, float, Tour], None]]


@dataclass
class HillClimbingConfig:
    """Hyper-parameters for the simple stochastic hill-climber."""

    n_iterations: int = 4000
    seed: int = 42


class HillClimbingTSP:
    """Random-restart-free hill climbing using 2-opt moves."""

    def __init__(self, D: np.ndarray, cfg: HillClimbingConfig):
        self.D = D.astype(float)
        self.n = D.shape[0]
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def run(self, on_iter: IterationCallback = None) -> AlgorithmResult:
        current = self._random_tour()
        current_length = tour_length(current, self.D)
        best = current[:]
        best_len = current_length
        history: History = [best_len]

        for iteration in range(self.cfg.n_iterations):
            candidate = self._two_opt_move(current)
            candidate_len = tour_length(candidate, self.D)

            if candidate_len < current_length:
                current, current_length = candidate, candidate_len
                if current_length < best_len:
                    best, best_len = current, current_length

            history.append(best_len)

            if on_iter:
                on_iter(iteration, best_len, best[:])

        return best, best_len, history

    def _random_tour(self) -> Tour:
        tour = list(range(self.n))
        self.rng.shuffle(tour)
        tour.append(tour[0])
        return tour

    def _two_opt_move(self, tour: Tour) -> Tour:
        n = len(tour) - 1
        i, j = sorted(self.rng.choice(range(1, n), size=2, replace=False))
        return tour[:i] + tour[i:j][::-1] + tour[j:]


def hill_climbing_tsp(
    D: np.ndarray,
    n_iterations: int = 4000,
    seed: int = 42,
    on_iter: IterationCallback = None,
) -> AlgorithmResult:
    """
    Backwards-compatible functional wrapper.
    """
    cfg = HillClimbingConfig(n_iterations=n_iterations, seed=seed)
    solver = HillClimbingTSP(D, cfg)
    return solver.run(on_iter)
