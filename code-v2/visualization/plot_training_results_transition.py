from typing import Tuple, List

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"font.size": 12, "figure.figsize": [9, 5]})


def read_results(log_path: str) -> Tuple[List[float], List[float], List[float]]:
    losses = []
    lens = []
    goals = []

    with open(log_path) as res:
        for line in res:
            ep = [float(x) for x in line.split(", ")]
            losses.append(ep[0])

    return losses


parser = argparse.ArgumentParser()
parser.add_argument("--labels", help="plot labels", nargs='+', type=str, required=True)
parser.add_argument("--paths", help="paths to the result directories", nargs='+', type=str, required=True)
parser.add_argument("--save-dir", help="path to the directory where to save plots", type=str, default=None)


if __name__ == "__main__":
    args = vars(parser.parse_args())

    data = []

    for dir in args["paths"]:
        i = 0
        log_path = Path(f"{dir}/run_{i}_log.txt")

        l_all = []

        while log_path.exists():
            losses = read_results(log_path)
            l_all.append(losses)

            i += 1
            log_path = Path(f"{dir}/run_{i}_log.txt")
        
        l_all = np.array(l_all)

        data.append({"avg": {"losses": l_all.mean(0)},
                     "std": {"losses": l_all.std(0)}})

    epochs = np.arange(max([len(d["avg"]["losses"]) for d in data]))

    for d in data:
        for stat in ("avg", "std"):
            if stat not in d:
                continue

            for metric in ("losses",):
                d[stat][metric] = np.pad(d[stat][metric],
                                         (0, len(epochs) - len(d[stat][metric])),
                                         constant_values=np.nan)

    fig, ax = plt.subplots()
    plt.get_current_fig_manager().set_window_title("Loss")

    for i, (l, d) in enumerate(zip(args["labels"], data)):
        ax.plot(epochs, d["avg"]["losses"], color=f"C{i}", label=l)
        
        if "std" in d:
            ax.fill_between(epochs,
                            d["avg"]["losses"] - d["std"]["losses"],
                            d["avg"]["losses"] + d["std"]["losses"],
                            color=f"C{i}", edgecolor=None, alpha=0.3)
        
    ax.set_xlabel("Epoch")
    ax.set_xticks(np.arange(0, len(epochs) + 1, 20))
    ax.set_xticks(np.arange(0, len(epochs) + 1, 10), minor=True)
    ax.set_xlim(0, len(epochs))
    ax.set_ylabel("Loss")
    bottom = -20
    top = 0
    ax.set_yticks(np.arange(bottom, top + 1, 2))
    ax.set_yticks(np.arange(bottom, top + 1), minor=True)
    ax.set_ylim(bottom, top)
    ax.legend(loc='best')
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(which="both")
    fig.tight_layout()
    
    if args["save_dir"]:
        fig.savefig(f'{args["save_dir"]}/loss.pdf')
    else:
        plt.show()
