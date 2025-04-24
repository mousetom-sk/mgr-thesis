from typing import Tuple, List

import argparse
from pathlib import Path

import numpy as np


def read_results(test_path: Path) -> Tuple[List[float], List[float]]:
    losses = []
    losses_std = []

    with open(test_path) as res:
        for line in res:
            ep = [float(x) for x in line.split(", ")]
            losses.append(ep[0])
            losses_std.append(ep[1])

    return losses, losses_std


parser = argparse.ArgumentParser()
parser.add_argument("--labels", help="table labels", nargs='+', type=str, required=True)
parser.add_argument("--paths", help="paths to the result directories", nargs='+', type=str, required=True)


if __name__ == "__main__":
    args = vars(parser.parse_args())

    data = []

    for dir in args["paths"]:
        test_path = Path(f"{dir}/test.txt")

        losses = read_results(test_path)[0]
        losses = np.array(losses)

        data.append({"avg": {"loss": losses.mean()},
                     "std": {"loss": losses.std()}})
    
    for l, d in zip(args["labels"], data):
        row = l
        row += f' & {d["avg"]["loss"]:.3f} & \\pm & {d["std"]["loss"]:.3f}'
        row += " \\\\"

        print(row)
