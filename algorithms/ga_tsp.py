# algorithms/ga_tsp.py
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from problems.tsp import tour_length

Tour = List[int]
History = List[float]
AlgorithmResult = Tuple[Tour, float, History]
IterationCallback = Optional[Callable[[int, float, Optional[Tour]], None]]


@dataclass
class GAConfig:
    """Hyper-parameters for the Genetic Algorithm solver."""

    pop_size: int = 260          # good defaults for 25–50 nodes
    n_gen: int = 800
    elite_ratio: float = 0.08    # keep top 8%
    crossover_rate: float = 0.92
    mutation_rate: float = 0.15
    tournament_k: int = 3
    seed: int = 123


class GeneticAlgorithmTSP:
    """Straight-forward GA optimized for closed TSP tours."""

    def __init__(self, D: np.ndarray, cfg: GAConfig):
        self.D = D.astype(float)
        self.n = D.shape[0]
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def run(self, on_iter: IterationCallback = None) -> AlgorithmResult:
        population = self._init_population()
        best: Optional[Tour] = None
        best_len = float("inf")
        history: History = []
        n_elite = max(1, int(self.cfg.elite_ratio * self.cfg.pop_size))

        for gen in range(self.cfg.n_gen):
            fits = np.array([self._fitness_of(t) for t in population])
            ibest = int(np.argmax(fits))
            L_best = 1.0 / fits[ibest]
            if L_best < best_len:
                best, best_len = population[ibest][:], float(L_best)

            elite_idx = np.argsort(fits)[-n_elite:]
            new_population: List[Tour] = [population[i][:] for i in elite_idx]

            while len(new_population) < self.cfg.pop_size:
                i1 = self._tournament_select_index(fits)
                i2 = self._tournament_select_index(fits)
                p1, p2 = population[i1], population[i2]

                if self.rng.random() < self.cfg.crossover_rate:
                    child = self._ox_crossover(p1, p2)
                else:
                    child = p1[:]

                child = self._mutate_swap(child)
                new_population.append(child)

            population = new_population
            history.append(best_len)

            if on_iter:
                best_copy = best[:] if best else None
                on_iter(gen, best_len, best_copy)

        assert best is not None
        return best, best_len, history

    def _init_population(self) -> List[Tour]:
        base = list(range(self.n))
        population: List[Tour] = []
        for _ in range(self.cfg.pop_size):
            self.rng.shuffle(base)
            population.append(base[:] + [base[0]])
        return population

    def _fitness_of(self, tour: Tour) -> float:
        L = tour_length(tour, self.D)
        return 1.0 / (1e-9 + L)

    def _tournament_select_index(self, fits: np.ndarray) -> int:
        idx = self.rng.integers(0, len(fits), size=self.cfg.tournament_k)
        return int(idx[np.argmax(fits[idx])])

    def _ox_crossover(self, p1: Tour, p2: Tour) -> Tour:
        n = len(p1) - 1
        a, b = sorted(self.rng.integers(1, n, size=2))
        child = [-1] * n
        child[a:b] = p1[a:b]
        fill = [g for g in p2[:n] if g not in child]
        pos = b
        for g in fill:
            if pos == n:
                pos = 0
            while child[pos] != -1:
                pos = (pos + 1) % n
            child[pos] = g
            pos = (pos + 1) % n
        child.append(child[0])
        return child

    def _mutate_swap(self, tour: Tour) -> Tour:
        if self.rng.random() < self.cfg.mutation_rate:
            n = len(tour) - 1
            i, j = sorted(self.rng.integers(1, n, size=2))
            tour[i], tour[j] = tour[j], tour[i]
            tour[-1] = tour[0]
        return tour


def ga_tsp(
    D: np.ndarray,
    pop_size: int = 260,
    n_gen: int = 800,
    elite_ratio: float = 0.08,
    crossover_rate: float = 0.92,
    mutation_rate: float = 0.15,
    tournament_k: int = 3,
    seed: int = 123,
    on_iter: IterationCallback = None,
) -> AlgorithmResult:
    """
    Convenience wrapper mirroring the legacy functional API.
    """
    cfg = GAConfig(
        pop_size=pop_size,
        n_gen=n_gen,
        elite_ratio=elite_ratio,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        tournament_k=tournament_k,
        seed=seed,
    )
    solver = GeneticAlgorithmTSP(D, cfg)
    return solver.run(on_iter)
