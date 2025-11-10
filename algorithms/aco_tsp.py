import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from problems.tsp import tour_length

Tour = List[int]
History = List[float]
AlgorithmResult = Tuple[Tour, float, History]
IterationCallback = Optional[Callable[[int, float, Optional[Tour], Dict[str, np.ndarray]], None]]

@dataclass
class ACOConfig:
    """Hyper-parameters for the Ant Colony Optimization solver."""

    n_ants: Optional[int] = None
    n_iterations: int = 200
    alpha: float = 1.0
    beta: float = 5.0
    rho: float = 0.45        # evaporation rate
    q: float = 75.0          # pheromone deposit factor
    elitist_weight: float = 0.5  # >0 để tăng cường best_global
    seed: int = 42


class AntColonyTSP:
    """Classic Ant Colony Optimization solver for the symmetric TSP."""

    def __init__(self, D: np.ndarray, cfg: ACOConfig):
        self.D = D.astype(float)
        self.n = D.shape[0]
        self.cfg = cfg
        # n_ants động nếu chưa set
        if self.cfg.n_ants is None:
            self.cfg.n_ants = self.n if self.n <= 20 else max(10, self.n // 3)

        self.rng = np.random.default_rng(cfg.seed)
        self.tau = np.ones((self.n, self.n), dtype=float)            # pheromone
        self.eta = 1.0 / (self.D + 1e-12)                            # heuristic
        np.fill_diagonal(self.eta, 0.0)

        self.best_tour: Optional[Tour] = None
        self.best_length: float = float("inf")

        # Ngưỡng clamp pheromone để tránh bùng nổ / tiêu biến
        self._tau_min, self._tau_max = 1e-9, 1e9

    def _select_next(self, i: int, unvisited) -> int:
        unv = np.fromiter(unvisited, dtype=int)
        numer = (self.tau[i, unv] ** self.cfg.alpha) * (self.eta[i, unv] ** self.cfg.beta)
        s = float(numer.sum())
        if s <= 0.0 or not np.isfinite(s):
            # fallback uniform
            return int(self.rng.choice(unv))
        probs = numer / s
        return int(self.rng.choice(unv, p=probs))

    def _build_tour(self, start: int) -> List[int]:
        tour = [start]
        unvisited = set(range(self.n))
        unvisited.remove(start)
        while unvisited:
            j = self._select_next(tour[-1], unvisited)
            tour.append(j)
            unvisited.remove(j)
        tour.append(start)
        return tour

    def _update_pheromones(self, tours: List[List[int]], lengths: List[float]) -> None:
        # Bay hơi
        self.tau *= (1.0 - self.cfg.rho)

        # Lắng đọng từ tất cả kiến
        for tour, L in zip(tours, lengths):
            dep = self.cfg.q / (L + 1e-12)
            for k in range(len(tour) - 1):
                a, b = tour[k], tour[k + 1]
                self.tau[a, b] += dep
                self.tau[b, a] += dep

        # Elitist: tăng cho best toàn cục (nếu bật)
        if self.cfg.elitist_weight > 0 and self.best_tour is not None:
            dep = self.cfg.elitist_weight * self.cfg.q / (self.best_length + 1e-12)
            for k in range(len(self.best_tour) - 1):
                a, b = self.best_tour[k], self.best_tour[k + 1]
                self.tau[a, b] += dep
                self.tau[b, a] += dep

        # Clamp để ổn định số học
        np.clip(self.tau, self._tau_min, self._tau_max, out=self.tau)

    def run(self, on_iter: IterationCallback = None) -> AlgorithmResult:
        n_ants = int(self.cfg.n_ants)
        history: History = []

        for it in range(self.cfg.n_iterations):
            tours: List[List[int]] = []
            lengths: List[float] = []

            # Mẹo: xoay vòng điểm start để đa dạng
            for k in range(n_ants):
                start = k % self.n  # hoặc: int(self.rng.integers(self.n))
                t = self._build_tour(start)
                L = tour_length(t, self.D)
                tours.append(t)
                lengths.append(L)

            # Cập nhật best toàn cục
            i_best = int(np.argmin(lengths))
            if lengths[i_best] < self.best_length:
                self.best_length = float(lengths[i_best])
                self.best_tour = tours[i_best]

            # Cập nhật pheromone
            self._update_pheromones(tours, lengths)

            # Lưu lịch sử best-so-far
            history.append(self.best_length)

            if on_iter:
                # Chụp snapshot tau mỗi k vòng để tiết kiệm RAM
                extras = {}
                if it % 2 == 0:  # hoặc %5 tuỳ bạn
                    extras["tau"] = self.tau.copy()
                best_copy = self.best_tour[:] if self.best_tour else None
                on_iter(it, self.best_length, best_copy, extras)
            

        assert self.best_tour is not None
        return self.best_tour, self.best_length, history
