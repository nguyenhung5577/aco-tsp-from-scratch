import matplotlib.pyplot as plt

def plot_convergence(histories: dict):
    plt.figure()
    for name, hist in histories.items():
        plt.plot(hist, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Best tour length")
    plt.legend()
    plt.title("So sánh hội tụ - TSP (ACO, SA, GA, HC)")
    plt.show()
