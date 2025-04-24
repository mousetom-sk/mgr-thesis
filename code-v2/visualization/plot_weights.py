import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({"font.size": 12, "figure.figsize": [15, 5.2]})


actions = ["move(b, a)", "move(c, a)", "move(d, a)", # "move(e, a)", "move(f, a)",
           "move(a, b)", "move(c, b)", "move(d, b)", # "move(e, b)", "move(f, b)",
           "move(a, c)", "move(b, c)", "move(d, c)", # "move(e, c)", "move(f, c)",
           "move(a, d)", "move(b, d)", "move(c, d)", # "move(e, d)", "move(f, d)",
           # "move(a, e)", "move(b, e)", "move(c, e)", "move(d, e)", "move(f, e)",
           # "move(a, f)", "move(b, f)", "move(c, f)", "move(d, f)", "move(e, f)",
           "move(a, table)", "move(b, table)", "move(c, table)", "move(d, table)", # "move(e, table)", "move(f, table)"
           ]

inputs = ["top(a)", "on(b, a)", "on(c, a)", "on(d, a)", # "on(e, a)", "on(f, a)",
          "top(b)", "on(a, b)", "on(c, b)", "on(d, b)", # "on(e, b)", "on(f, b)",
          "top(c)", "on(a, c)", "on(b, c)", "on(d, c)", # "on(e, c)", "on(f, c)",
          "top(d)", "on(a, d)", "on(b, d)", "on(c, d)", # "on(e, d)", "on(f, d)",
          # "top(e)", "on(a, e)", "on(b, e)", "on(c, e)", "on(d, e)", "on(f, e)",
          # "top(f)", "on(a, f)", "on(b, f)", "on(c, f)", "on(d, f)", "on(e, f)",
          "on(a, table)", "on(b, table)", "on(c, table)", "on(d, table)", # "on(e, table)", "on(f, table)",
        #   "$\\bot$"
          ]

ginputs = ["top_goal(a)", "on_goal(b, a)", "on_goal(c, a)", "on_goal(d, a)",
           "top_goal(b)", "on_goal(a, b)", "on_goal(c, b)", "on_goal(d, b)",
           "top_goal(c)", "on_goal(a, c)", "on_goal(b, c)", "on_goal(d, c)",
           "top_goal(d)", "on_goal(a, d)", "on_goal(b, d)", "on_goal(c, d)",
           "on_goal(a, table)", "on_goal(b, table)", "on_goal(c, table)", "on_goal(d, table)"]

for i in range(len(ginputs)):
    inputs.insert(2 * i + 1, ginputs[i])

# negated_inputs = [f"\\dneg {i}" for i in inputs]
# negated_inputs[-1] = "\\top"

transforms = {
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid
}

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="path to the result directory", type=str, required=True)
parser.add_argument("--run", help="the run whose weights to plot", type=int, default=None)
parser.add_argument("--epochs", help="epochs to plot", nargs='+', type=int, required=True)
parser.add_argument("--transform", help="how to transform the weights", choices=transforms, required=True)
parser.add_argument("--save-dir", help="path to the directory where to save plots", type=str, default=None)


if __name__ == "__main__":
    args = vars(parser.parse_args())

    data = {}
    trans = transforms[args["transform"]]

    for ep in args["epochs"]:
        i = 0 if args["run"] is None else args["run"]
        weights_path = Path(f'{args["dir"]}/run_{i}_ep_{ep}_actor.weights')
        data[ep] = {a: {} for a in actions}

        while weights_path.exists():
            with open(weights_path) as res:
                reading = False
                was_or = False
                is_and = False
                weight = []
                j = -1
                k = -1
                
                for line in res:
                    if "And" in line:
                        is_and = True
                        j += not was_or

                        if was_or:
                            k += 1
                        else:
                            k = 0
                    elif "Or" in line:
                        is_and = False
                        was_or = True
                        j += 1
                        k = -1

                    if reading:
                        if "]" in line:
                            weight.extend([trans(torch.tensor(float(w.rstrip(","))))
                                           for w in line.split("]")[0].split()])
                            if k not in data[ep][actions[j]]:
                                data[ep][actions[j]][k] = []
                            data[ep][actions[j]][k].append(weight)

                            reading = False
                            weight = []
                            continue

                        weight.extend([trans(torch.tensor(float(w.rstrip(","))))
                                       for w in line.split()])
                        
                    if is_and and "tensor" in line:
                        weight.extend([trans(torch.tensor(float(w.rstrip(","))))
                                       for w in line.split("[")[1].split()])
                        reading = True

            if args["run"] is not None:
                break

            i += 1
            weights_path = Path(f'{args["dir"]}/run_{i}_ep_{ep}_actor.weights')

    for a in actions:
        fig, ax = plt.subplots()
        plt.get_current_fig_manager().set_window_title(f"Learned Weights for Action {a}")
        x = np.arange(len(inputs))
        width = 1 / (len(args["epochs"]) + 1)

        if args["run"] is None:
            for i, ep in enumerate(args["epochs"]):
                avg = np.array(data[ep][a][0]).mean(0)
                std = np.array(data[ep][a][0]).std(0)

                ax.bar(x + i * width, avg, yerr=std, width=width, label=f"Epoch {ep}",
                    error_kw={"ecolor": "dimgray", "capsize": 2, "barsabove": True})
        else:
            for i, ep in enumerate(args["epochs"]):
                y = data[ep][a][0][0]
                x = x[:len(y)]

                ax.bar(x + i * width, y, width=width, label=f"Epoch {ep}")

        ax.set_xlabel("Input")
        ax.set_xticks(x + width * (len(args["epochs"]) // 2), inputs[:len(x)], rotation=30, ha="right")
        ax.set_ylabel("Transformed Weight")
        bottom = -1
        top = 1
        ax.set_yticks(np.arange(10 * bottom, 10 * top + 1, 2) / 10)
        ax.set_ylim(bottom, top)
        ax.legend(loc='best') # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), fancybox=True, ncol=len(args["epochs"]))
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", linestyle="--")
        fig.tight_layout()

        if args["save_dir"]:
            transl = dict.fromkeys(map(ord, ",("), "_") | {ord(" "): "", ord(")"): ""}
            fig.savefig(f'{args["save_dir"]}/{a.translate(transl)}.pdf')
        else:
            plt.show()
