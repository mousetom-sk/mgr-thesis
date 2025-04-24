from typing import Tuple, List

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"font.size": 12, "figure.figsize": [9, 5]})


def read_results(log_path: str) -> Tuple[List[float], List[float], List[float]]:
    returns = []
    lens = []
    goals = []

    with open(log_path) as res:
        for line in res:
            ep = [float(x) for x in line.split(", ")]
            returns.append(ep[0])
            lens.append(ep[2])
            goals.append(ep[4])

    period = 10
    returns_moving = []
    lens_moving = []
    goals_moving = []

    for i in range(len(returns)):
        returns_moving.append(np.array(returns[max(i -period + 1, 0):i + 1]).mean())
        lens_moving.append(np.array(lens[max(i - period + 1, 0):i + 1]).mean())
        goals_moving.append(np.array(goals[max(i - period + 1, 0):i + 1]).mean())

    return returns_moving, lens_moving, goals_moving


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

        if not log_path.exists():
            log_path = Path(f"{dir}/log.txt")
            returns, lens, goals = read_results(log_path)

            data.append({"avg": {"returns": returns, "lens": lens, "goals": goals}})
        else:
            r_all, l_all, g_all = [], [], []

            while log_path.exists():
                returns, lens, goals = read_results(log_path)
                r_all.append(returns)
                l_all.append(lens)
                g_all.append(goals)

                i += 1
                log_path = Path(f"{dir}/run_{i}_log.txt")
            
            r_all = np.array(r_all)
            l_all = np.array(l_all)
            g_all = np.array(g_all)

            data.append({"avg": {"returns": r_all.mean(0),
                                 "lens": l_all.mean(0),
                                 "goals": g_all.mean(0)},
                         "std": {"returns": r_all.std(0),
                                 "lens": l_all.std(0),
                                 "goals": g_all.std(0)}})
    
    epochs = np.arange(max([len(d["avg"]["returns"]) for d in data]))

    for d in data:
        for stat in ("avg", "std"):
            if stat not in d:
                continue

            for metric in ("returns", "lens", "goals"):
                d[stat][metric] = np.pad(d[stat][metric],
                                         (0, len(epochs) - len(d[stat][metric])),
                                         constant_values=np.nan)

    fig, ax = plt.subplots()
    plt.get_current_fig_manager().set_window_title("Episode Length")

    for i, (l, d) in enumerate(zip(args["labels"], data)):
        ax.plot(epochs, d["avg"]["lens"], color=f"C{i}", label=l)
        
        if "std" in d:
            ax.fill_between(epochs,
                            d["avg"]["lens"] - d["std"]["lens"],
                            d["avg"]["lens"] + d["std"]["lens"],
                            color=f"C{i}", edgecolor=None, alpha=0.3)
        
    ax.set_xlabel("Epoch")
    ax.set_xticks(np.arange(0, len(epochs) + 1, 20))
    ax.set_xticks(np.arange(0, len(epochs) + 1, 10), minor=True)
    ax.set_xlim(0, len(epochs))
    ax.set_ylabel("Episode Length")
    bottom = 0
    top = 5000
    ax.set_yticks(np.arange(bottom, top + 1, 500))
    ax.set_yticks(np.arange(bottom, top + 1, 100), minor=True)
    ax.set_ylim(bottom, top)
    ax.legend(loc='best')
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(which="both")
    fig.tight_layout()
    
    if args["save_dir"]:
        fig.savefig(f'{args["save_dir"]}/length.pdf')
    else:
        plt.show()

    fig, ax = plt.subplots()
    plt.get_current_fig_manager().set_window_title("Success Rate")

    for i, (l, d) in enumerate(zip(args["labels"], data)):
        ax.plot(epochs, d["avg"]["goals"], color=f"C{i}", label=l)
        
        if "std" in d:
            ax.fill_between(epochs,
                            d["avg"]["goals"] - d["std"]["goals"],
                            d["avg"]["goals"] + d["std"]["goals"],
                            color=f"C{i}", edgecolor=None, alpha=0.3)
        
    ax.set_xlabel("Epoch")
    ax.set_xticks(np.arange(0, len(epochs) + 1, 20))
    ax.set_xticks(np.arange(0, len(epochs) + 1, 10), minor=True)
    ax.set_xlim(0, len(epochs))
    ax.set_ylabel("Success Rate")
    ax.set_yticks(np.arange(0, 11) / 10)
    ax.set_yticks(np.arange(0, 101, 5) / 100, minor=True)
    ax.set_ylim(0, 1.01)
    ax.legend(loc='best')
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(which="both")
    fig.tight_layout()
    
    if args["save_dir"]:
        fig.savefig(f'{args["save_dir"]}/success.pdf')
    else:
        plt.show()
