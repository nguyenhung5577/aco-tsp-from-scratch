import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from problems.tsp import tour_length

Tour = List[int]
History = List[float]
AlgorithmResult = Tuple[Tour, float, History]
IterationCallback = Optional[Callable[[int, float, Tour], None]]


@dataclass
class SAConfig:
    """Hyper-parameters for the Simulated Annealing solver."""

    n_iterations: int = 3000
    alpha: float = 0.998
    steps_per_T: int = 30
    T0: Optional[float] = None          # None -> tự ước lượng từ ΔE dương
    seed: int = 123


class SimulatedAnnealingTSP:
    """Classic SA with 2-opt neighborhood for the TSP."""

    def __init__(self, D: np.ndarray, cfg: SAConfig):
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

        temperature = (
            self._estimate_T0(current) if self.cfg.T0 is None else float(self.cfg.T0)
        )

        for it in range(self.cfg.n_iterations):
            for _ in range(self.cfg.steps_per_T):
                candidate = self._two_opt_move(current)
                candidate_len = tour_length(candidate, self.D)
                delta = candidate_len - current_length

                if delta < 0 or self._accept(delta, temperature):
                    current, current_length = candidate, candidate_len
                    if current_length < best_len:
                        best, best_len = current, current_length

            temperature *= self.cfg.alpha
            history.append(best_len)

            if on_iter:
                on_iter(it, best_len, best[:])

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

    def _accept(self, delta: float, temperature: float) -> bool:
        return self.rng.random() < np.exp(-min(700.0, delta / max(1e-12, temperature)))

    def _estimate_T0(self, tour: Tour, trials: int = 100, p0: float = 0.8) -> float:
        """Ước lượng T0 sao cho P(accept ΔE>0) ≈ p0 ở đầu quá trình."""
        n = len(tour) - 1
        L0 = tour_length(tour, self.D)
        positive_jumps = []
        for _ in range(trials):
            i, j = sorted(self.rng.choice(range(1, n), size=2, replace=False))
            new = tour[:i] + tour[i:j][::-1] + tour[j:]
            dE = tour_length(new, self.D) - L0
            if dE > 0:
                positive_jumps.append(dE)
        if not positive_jumps:
            return max(1.0, 0.1 * L0)
        mean_dE = float(np.mean(positive_jumps))
        return max(1e-9, -mean_dE / np.log(p0))


def simulated_annealing_tsp(
    D: np.ndarray,
    n_iterations: int = 3000,
    T0: Optional[float] = None,
    alpha: float = 0.998,
    seed: int = 123,
    steps_per_T: int = 30,
    on_iter: IterationCallback = None,
) -> AlgorithmResult:
    """
    Convenience wrapper that preserves the historical functional signature.
    """
    cfg = SAConfig(
        n_iterations=n_iterations,
        alpha=alpha,
        steps_per_T=steps_per_T,
        T0=T0,
        seed=seed,
    )
    solver = SimulatedAnnealingTSP(D, cfg)
    return solver.run(on_iter)
