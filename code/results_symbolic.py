import numpy as np


models = [
    {"label": "LMLP AC", "dir": "models/lmlp_a2c"},
    {"label": "LMLP AC Regularized", "dir": "models/lmlp_a2c_reg"},
    {"label": "Symbolic Planning", "dir": "models/planning"}
]


if __name__ == "__main__":
    data = []
    
    for m in models:
        with open(f'{m["dir"]}/test_results.txt') as res:
            returns = [float(n) for n in res.readline().split(", ")]
            goals = [float(n) for n in res.readline().split(", ")]
            data.append({"returns": np.array(returns), "goals": np.array(goals)})

    for m, d in zip(models, data):
        print(f'{m["label"]} & ${np.mean(d["returns"]):.3f} \\pm {np.std(d["returns"]):.3f}$ & ${np.mean(d["goals"]):.3f} \\pm {np.std(d["goals"]):.3f}$ \\\\')
