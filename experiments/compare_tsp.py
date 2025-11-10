import os, numpy as np
from problems.tsp import read_weight_matrix
from algorithms.aco_tsp import AntColonyTSP, ACOConfig
from algorithms.sa_tsp import simulated_annealing_tsp
from algorithms.ga_tsp import ga_tsp
from algorithms.hc_tsp import hill_climbing_tsp
from utils.plot import plot_convergence

def main():
    D = read_weight_matrix("data/weights.csv")

    # ACO
    cfg = ACOConfig(n_iterations=200, rho=0.5, beta=5.0)
    aco = AntColonyTSP(D, cfg)
    aco_tour, aco_len, aco_hist = aco.run()

    # SA
    sa_tour, sa_len, sa_hist = simulated_annealing_tsp(D, n_iterations=3000)

    # GA
    ga_tour, ga_len, ga_hist = ga_tsp(D, pop_size=300, n_gen=900)

    # HC
    hc_tour, hc_len, hc_hist = hill_climbing_tsp(D, n_iterations=3000)

    print("\n=== KẾT QUẢ ===")
    print(f"ACO: {aco_len:.2f}")
    print(f"SA : {sa_len:.2f}")
    print(f"GA : {ga_len:.2f}")
    print(f"HC : {hc_len:.2f}")

    print(aco_tour)
    print(sa_tour)
    print(ga_tour)
    print(hc_tour)

    plot_convergence({
        "ACO": aco_hist,
        "SA": sa_hist,
        "GA": ga_hist,
        "HC": hc_hist
    })



if __name__ == "__main__":
    main()
