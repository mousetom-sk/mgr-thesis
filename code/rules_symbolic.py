import numpy as np


action = "move(b, a)"
inputs = ["top(a)", "on(b, a)", "on(c, a)", "on(d, a)",
          "top(b)", "on(a, b)", "on(c, b)", "on(d, b)",
          "top(c)", "on(a, c)", "on(b, c)", "on(d, c)",
          "top(d)", "on(a, d)", "on(b, d)", "on(c, d)",
          "on(a, table)", "on(b, table)", "on(c, table)", "on(d, table)",
          "\\bot"]
models = [
    {"label": "LMLP AC", "dir": "models/lmlp_a2c"},
    {"label": "LMLP AC Regularized", "dir": "models/lmlp_a2c_reg"}
]


if __name__ == "__main__":
    data = []
    
    for m in models:
        with open(f'{m["dir"]}/weights.txt') as res:
            weight_count = 0
            reading = False
            weight = []
            
            for line in res:
                if "weight:" in line:
                    weight_count += 1

                if reading:
                    if "device" in line:
                        weight.extend([np.tanh(float(w[:-3])) for w in line.split() if "]]" in w])
                        break

                    weight.extend([np.tanh(float(w[:-1])) for w in line.split()])
                    
                if weight_count == 2 and "tensor" in line:
                    weight.extend([np.tanh(float(w[:-1])) for w in line.split()[1:]])
                    reading = True

        data.append(weight)

    for m, d in zip(models, data):
        print(m["label"])
        print(f'{action} \\leftarrow', ", ".join(f'{"\\lnot " if d[i] < 0 else ""}{v}^{{({abs(d[i]):.3f})}}' for i, v in enumerate(inputs) if abs(d[i]) > 3e-1))
        print()
