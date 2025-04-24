from typing import Tuple, List

import argparse
from pathlib import Path

import numpy as np


def read_results(test_path: Path) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    returns = []
    lens = []
    goals = []
    returns_std = []
    lens_std = []
    goals_std = []

    with open(test_path) as res:
        for line in res:
            ep = [float(x) for x in line.split(", ")]
            returns.append(ep[0])
            lens.append(ep[2])
            goals.append(ep[4])
            returns_std.append(ep[1])
            lens_std.append(ep[3])
            goals_std.append(ep[5])

    return returns, lens, goals, returns_std, lens_std, goals_std


parser = argparse.ArgumentParser()
parser.add_argument("--labels", help="table labels", nargs='+', type=str, required=True)
parser.add_argument("--paths", help="paths to the result directories", nargs='+', type=str, required=True)


if __name__ == "__main__":
    args = vars(parser.parse_args())

    data = []

    for dir in args["paths"]:
        test_path = Path(f"{dir}/test.txt")

        returns, lens, goals, returns_std, lens_std, goals_std = read_results(test_path)
        
        if len(returns) > 1:
            r_all, l_all, g_all = returns, lens, goals
            
            r_all = np.array(r_all)
            l_all = np.array(l_all)
            g_all = np.array(g_all)

            data.append({"avg": {"return": r_all.mean(),
                                 "len": l_all.mean(),
                                 "goal": g_all.mean()},
                         "std": {"return": r_all.std(),
                                 "len": l_all.std(),
                                 "goal": g_all.std()}})
        else:
            data.append({"avg": {"return": returns[0],
                                 "len": lens[0],
                                 "goal": goals[0]},
                         "std": {"return": returns_std[0],
                                 "len": lens_std[0],
                                 "goal": goals_std[0]}})
    
    for l, d in zip(args["labels"], data):
        row = l
        row += f' & {d["avg"]["return"]:.3f} & \\pm & {d["std"]["return"]:.3f}'
        row += f' & {d["avg"]["len"]:.3f} & \\pm & {d["std"]["len"]:.3f}'
        row += f' & {d["avg"]["goal"]:.3f} & \\pm & {d["std"]["goal"]:.3f}'
        row += " \\\\"

        print(row)
