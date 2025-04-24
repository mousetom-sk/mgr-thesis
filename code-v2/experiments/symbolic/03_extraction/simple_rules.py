from typing import List, Dict, Set

import argparse

import torch
from torch import Tensor

from nesyrl.envs.symbolic import BlocksWorld, On, Top, Contradiction, PredicateAtom


def atom_to_str(atom: PredicateAtom, negated: bool, use_tex: bool) -> str:
    res = str(atom)
    
    if negated:
        if isinstance(atom, Contradiction):
            res = "true"
        else:
            res = f"not {res}"
    
    if use_tex:
        res = (res.replace("move", "\\amove")
                  .replace("table", "\\ctable")
                  .replace("top", "\\stop")
                  .replace("on", "\\son")
                  .replace("table", "\\ctable")
                  .replace("contradiction", "\\bot")
                  .replace("not", "\\dneg")
                  .replace("true", "\\top"))
        
    return res

def negate_str(atom_str: str, use_tex: bool) -> str:
    if use_tex:
        if atom_str.startswith("\\dneg"):
            return atom_str.replace("\\dneg ", "")
        
        return f"\\dneg {atom_str}"
    
    if atom_str.startswith("not"):
            return atom_str.replace("not ", "")
        
    return f"not {atom_str}"

def read_weights(actions: List[str], path: str) -> Dict[str, List[Tensor]]:
    data = {a: [] for a in actions}
    
    with open(path) as res:
        reading = False
        was_or = False
        is_and = False
        weight = []
        j = -1
        
        for line in res:
            if "And" in line:
                is_and = True
                j += not was_or
            elif "Or" in line:
                is_and = False
                was_or = True
                j += 1
        
            if reading:
                if "]" in line:
                    weight.extend([torch.tanh(torch.tensor(float(w.rstrip(","))))
                                   for w in line.split("]")[0].split()])
                    data[actions[j]].append(weight)

                    reading = False
                    weight = []
                    continue

                weight.extend([torch.tanh(torch.tensor(float(w.rstrip(","))))
                               for w in line.split()])
                
            if is_and and "tensor" in line:
                weight.extend([torch.tanh(torch.tensor(float(w.rstrip(","))))
                               for w in line.split("[")[1].split()])
                reading = True

    return data

def simplify_weight(weight: Tensor) -> Tensor:
    is_pos = weight > 0.7
    is_neg = weight < -0.7
    is_zero = ~is_pos & ~is_neg

    pos = torch.where(is_pos, torch.ones_like(weight), torch.inf)
    neg = torch.where(is_neg, -torch.ones_like(weight), torch.inf)
    zero = torch.where(is_zero, -torch.zeros_like(weight), torch.inf)

    return torch.minimum(pos, torch.minimum(neg, zero))

def prepare_disjoints(env: BlocksWorld, use_tex: bool) -> List[List[str]]:
    disjoints = []
    
    for b1 in env._blocks:
        disjoints.append([atom_to_str(On(b1, env._table), False, use_tex)]
                         + [atom_to_str(On(b1, b2), False, use_tex)
                            for b2 in env._blocks if b2 != b1])
        disjoints.append([atom_to_str(Top(b1), False, use_tex)]
                         + [atom_to_str(On(b2, b1), False, use_tex)
                            for b2 in env._blocks if b2 != b1])
        
    return disjoints

def simplify_rule(rule: Set[str], disjoints: List[List[str]], use_tex: bool) -> Set[str]:
    simplified = set(rule)
    
    for dis in disjoints:
        is_in = [lit in rule for lit in dis]
        num_in = sum(is_in)

        if num_in == 0:
            continue

        if num_in > 1:
            return set(atom_to_str(Contradiction(), False, use_tex))
        
        i = is_in.index(True)
        for j in range(len(dis)):
            if j == i:
                continue

            simplified.discard(negate_str(dis[j], use_tex))

    return simplified


parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="path to the result directory", type=str, required=True)
parser.add_argument("--run", help="the run whose weights to consider", type=int, required=True)
parser.add_argument("--blocks", help="the blocks to use", nargs="+", type=str, required=True)
parser.add_argument("--use-tex", help="whether the simplified rules shoud be TeX-formatted", action="store_true")

if __name__ == "__main__":
    args = vars(parser.parse_args())

    env = BlocksWorld(0, args["blocks"], [args["blocks"]])
    actions = [atom_to_str(a, False, args["use_tex"])
               for a in env.action_atoms]
    inputs = [atom_to_str(a, False, args["use_tex"])
              for a in env.state_atoms if a not in env.domain_atoms]
    negated_inputs = [atom_to_str(a, True, args["use_tex"])
                      for a in env.state_atoms if a not in env.domain_atoms]
    
    path = f'{args["dir"]}/run_{args["run"]}_actor.weights'
    weights = read_weights(actions, path)
    disjoints = prepare_disjoints(env, args["use_tex"])

    for a in weights:
        for w in weights[a]:
            w = simplify_weight(w)
            rule = set()

            for i in range(len(inputs)):
                if w[i] > 0:
                    rule.add(inputs[i])
                elif w[i] < 0:
                    rule.add(negated_inputs[i])

            simplified = simplify_rule(rule, disjoints, args["use_tex"])

            if args["use_tex"]:
                print(f'{a} \\leftarrow', ", ".join(simplified), "\\\\")
            else:
                print(f'{a} :-', ", ".join(simplified))
