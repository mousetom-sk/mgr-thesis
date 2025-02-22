import math
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 12})

models = [
    {"label": "LMLP AC Test", "dir": "models/lmlp_a2c_test"}
]


if __name__ == "__main__":
    data = []

    for m in models:
        with open(f'{m["dir"]}/probs.txt') as res:
            probs = [float(n) for n in res.readline().split(", ")]
            data.append({"probs": probs})
    
    times = np.arange(1, len(data[0]["probs"]) + 1)

    fig, ax = plt.subplots()
    plt.get_current_fig_manager().set_window_title("Max Action Probability")

    for m, d in zip(models, data):
        ax.plot(times, d["probs"], label=m["label"])

    ax.set_xlabel("Step")
    ax.set_xticks([1] + list(range(500, len(times) + 1, 500)))
    ax.set_xlim(1, len(times))
    ax.set_ylabel("Max Action Probability")
    bottom = 0
    top = 1
    ax.set_yticks(np.arange(10 * bottom, 10 * top + 1, 2) / 10)
    ax.set_ylim(bottom, top)
    ax.legend(loc="lower right")
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid()
    fig.tight_layout()
    plt.show()
