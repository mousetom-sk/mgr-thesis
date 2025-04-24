import argparse

import torch
import numpy as np


actions = ["\\amove(b, a)", "\\amove(c, a)", "\\amove(d, a)", "\\amove(e, a)",
           "\\amove(a, b)", "\\amove(c, b)", "\\amove(d, b)", "\\amove(e, b)",
           "\\amove(a, c)", "\\amove(b, c)", "\\amove(d, c)", "\\amove(e, c)",
           "\\amove(a, d)", "\\amove(b, d)", "\\amove(c, d)", "\\amove(e, d)",
           "\\amove(a, e)", "\\amove(b, e)", "\\amove(c, e)", "\\amove(d, e)",
           "\\amove(a, \\ctable)", "\\amove(b, \\ctable)", "\\amove(c, \\ctable)", "\\amove(d, \\ctable)", "\\amove(e, \\ctable)"
           ]

inputs = ["\\stop(a)", "\\son(b, a)", "\\son(c, a)", "\\son(d, a)", "\\son(e, a)",
          "\\stop(b)", "\\son(a, b)", "\\son(c, b)", "\\son(d, b)", "\\son(e, b)",
          "\\stop(c)", "\\son(a, c)", "\\son(b, c)", "\\son(d, c)", "\\son(e, c)",
          "\\stop(d)", "\\son(a, d)", "\\son(b, d)", "\\son(c, d)", "\\son(e, d)",
          "\\stop(e)", "\\son(a, e)", "\\son(b, e)", "\\son(c, e)", "\\son(d, e)",
          "\\son(a, \\ctable)", "\\son(b, \\ctable)", "\\son(c, \\ctable)", "\\son(d, \\ctable)", "\\son(e, \\ctable)",
          "\\bot"]

negated_inputs = [f"\\dneg {i}" for i in inputs]
negated_inputs[-1] = "\\top"

transforms = {
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid
}

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="path to the result directory", type=str, required=True)
parser.add_argument("--run", help="the run whose weights to consider", type=int, required=True)
parser.add_argument("--transform", help="how to transform the weights", choices=transforms, required=True)


if __name__ == "__main__":
    args = vars(parser.parse_args())

    data = []
    trans = transforms[args["transform"]]
    
    with open(f'{args["dir"]}/run_{args["run"]}_actor.weights') as res:
        reading = False
        was_or = False
        is_and = False
        weight = []
        
        for line in res:
            if "And" in line:
                is_and = True
            elif "Or" in line:
                is_and = False
                was_or = True
                data.append([])

            if reading:
                if "]" in line:
                    weight.extend([trans(torch.tensor(float(w.rstrip(","))))
                                   for w in line.split("]")[0].split()])
                    if was_or:
                        data[-1].append(weight)
                    else:
                        data.append(weight)

                    reading = False
                    weight = []
                    continue

                weight.extend([trans(torch.tensor(float(w.rstrip(","))))
                               for w in line.split()])
                
            if is_and and "tensor" in line:
                weight.extend([trans(torch.tensor(float(w.rstrip(","))))
                               for w in line.split("[")[1].split()])
                reading = True

    for a, dl in zip(actions, data):
        if not isinstance(dl[0], list):
            dl = [dl]

        for d in dl:
            if len(d) == len(inputs):
                print(f'{a} \\leftarrow', ", ".join(
                    f'{negated_inputs[i] if d[i] < 0 else inputs[i]}^{{({abs(d[i]):.3f})}}'
                    for i in range(len(d)) if abs(d[i]) > 1e-1
                ), "\\\\")
            else:
                print(f'{a} \\leftarrow', ", ".join(
                    f'{negated_inputs[i % len(inputs)] if d[i] < 0 else inputs[i]}^{{({abs(d[i]):.3f})}}'
                    for i in range(len(d)) if abs(d[i]) > 1e-1
                ), "\\\\")
