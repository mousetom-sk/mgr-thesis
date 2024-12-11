import math
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 12})

models = [
    {"label": "LMLP AC", "dir": "models/lmlp_a2c"},
    {"label": "LMLP AC Regularized", "dir": "models/lmlp_a2c_reg"},
    {"label": "Symbolic Planning", "dir": "models/planning"}
]


if __name__ == "__main__":
    data = []
    min_return = max_return = 0

    for m in models:
        with open(f'{m["dir"]}/training_results.txt') as res:
            returns = [float(n) for n in res.readline().split(", ")]
            goals = [float(n) for n in res.readline().split(", ")]
            data.append({"returns": returns, "goals": goals})
            min_return = min(min_return, min(returns))
            max_return = max(max_return, max(returns))
    
    times = np.arange(1, len(data[0]["returns"]) + 1)

    fig, ax = plt.subplots()
    plt.get_current_fig_manager().set_window_title("Average Return")

    for m, d in zip(models, data):
        ax.plot(times, d["returns"], label=m["label"])

    ax.set_xlabel("Evaluation Time")
    ax.set_xticks([1] + list(range(10, len(times) + 1, 10)))
    ax.set_xlim(1, len(times))
    ax.set_ylabel("Average Return")
    bottom = math.floor(min_return)
    top = math.ceil(max_return)
    ax.set_yticks(np.arange(10 * bottom, 10 * top + 1, 2) / 10)
    ax.set_ylim(bottom, top)
    ax.legend(loc="lower right")
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid()
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    plt.get_current_fig_manager().set_window_title("Success Rate")

    for m, d in zip(models, data):
        ax.plot(times, d["goals"], label=m["label"])
        
    ax.set_xlabel("Evaluation Time")
    ax.set_xticks([1] + list(range(10, len(times) + 1, 10)))
    ax.set_xlim(1, len(times))
    ax.set_ylabel("Success Rate")
    ax.set_yticks(np.arange(0, 11) / 10)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid()
    fig.tight_layout()
    plt.show()
