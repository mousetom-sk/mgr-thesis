from typing import List, Tuple

import time
import argparse
import json
from pathlib import Path

import torch
from torch.optim import Optimizer

from nesyrl.envs.symbolic import BlocksWorld
from nesyrl.agents.symbolic import TransitionModel
from nesyrl.logic.neural import ConstantInitializer, UniformInitializer, NLAndBiProd, NLAndBiLuka
from nesyrl.logic.propositional import Contradiction, Valuation


def load_transitions(path: str) -> List[Tuple[Valuation, Valuation, int]]:
    data = []

    with open(path) as src:
        head = src.readline()
        keys = [k.strip() for k in head.split(";")]
        l = (len(keys) - 1) // 2

        for tr in src.readlines():
            tr = tr.split(";")
            obs = {k: float(v) for k, v in zip(keys[:l], tr[:l])}
            obs_next = {k: float(v) for k, v in zip(keys[l:-1], tr[l:-1])}
            action = int(tr[-1])

            data.append((obs, obs_next, action))

    return data

def save_run(
    run: int, ep: int | None, model: TransitionModel, optim: Optimizer
) -> None:
    ep_str = f"_ep_{ep}" if ep is not None else ""

    torch.save(model, f"{log_dir}/run_{run}{ep_str}.model")
    torch.save(optim, f"{log_dir}/run_{run}{ep_str}.optim")

    with open(f"{log_dir}/run_{run}{ep_str}.weights", "w") as out:
        print(model.params_str(), file=out)

def calculate_loss(model: TransitionModel, obs: Valuation, obs_next: Valuation, action: int) -> torch.Tensor:
    obs_next_pred = model.forward(obs, action)
    obs_next = torch.tensor([obs_next[str(a)] for a in state_atoms]).to(obs_next_pred)

    neg_corr = - (obs_next * obs_next_pred + (1 - obs_next) * (1 - obs_next_pred))

    return torch.sum(neg_corr)

def test_model(
    model: TransitionModel, steps: int, env: BlocksWorld, path: str
) -> None:
    model.eval()

    losses = []
    
    for _ in range(steps):
        obs, obs_next, action = env.generate_random_transition()
        losses.append(calculate_loss(model, obs, obs_next, action))

    model.train()

    losses = torch.tensor(losses)
    mean, std = losses.mean(), losses.std()

    with open(path, "a") as out:
        print(f"{float(mean)}, {float(std)}", file=out)

def next_epoch(
    ep: int, run: int, model: TransitionModel, optim: Optimizer,
    test_steps: int, test_env: BlocksWorld, test_path: str
) -> None:
    if ep % 10 == 0:
        save_run(run, ep, model, optim)
    
    test_model(model, test_steps, test_env, test_path)


# Basic configuration
initializers = {
    "constant": ConstantInitializer,
    "uniform": UniformInitializer
}

ands = {
    "biprod": NLAndBiProd,
    "biluka": NLAndBiLuka,
}

optimizers = {
    "rms": torch.optim.RMSprop,
    "adam": torch.optim.Adam
}

parser = argparse.ArgumentParser()
parser.add_argument("--train-dir", help="the directory with the collected transitions for training", type=str, required=True)
parser.add_argument("--init", help="the NLN weight initializer", choices=initializers, required=True)
parser.add_argument("--init-arg", help="argument for the NLN weight initializer", type=float, required=True)
parser.add_argument("--and", help="the implementation of conjuction to use", choices=ands, required=True)
parser.add_argument("--optim", help="optimizer of the transition model's parameters", choices=optimizers, required=True)
parser.add_argument("--lr", help="learning rate for the transition model's optimizer", type=float, required=True)

log_dir = f"results/symbolic/08_effects/nln_transition_model_{time.strftime('%d%m%Y_%H%M%S', time.gmtime(time.time()))}"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

args = {
    "test_env_kwargs":  {
        "horizon": 50,
        "blocks": ["a", "b", "c", "d"],
        "goal_state": [["a", "b", "c", "d"]]
    },
    "test_train_seed": 42,
    "test_test_seed": 47,
    "test_steps": 1000,
    "trainer": {
        "max_epoch": 300,
        "step_per_epoch": 100,
        "step_per_test": 100
    }
}


if __name__ == "__main__":
    args |= vars(parser.parse_args())

    # Prepare log directory
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    with open(f"{log_dir}/args.txt", "w") as out:
        json.dump(args, out, indent=4)

    run = 0
    trans_path = Path(f"{args['train_dir']}/run_{run}_transitions.txt")

    while trans_path.exists():
        # Prepare environments
        test_env = BlocksWorld(**args["test_env_kwargs"])
        test_env.reset(seed=args["test_train_seed"])

        # Prepare transition model
        model = TransitionModel(
            test_env,
            ands[args["and"]],
            initializers[args["init"]](args["init_arg"]),
            device=device
        )
        
        optim = optimizers[args["optim"]](model.parameters(), lr=args["lr"])

        # Prepare training
        training_data = iter(load_transitions(trans_path))
        state_atoms = [atom for atom in test_env.state_atoms
                       if atom not in test_env.domain_atoms
                       and not isinstance(atom, Contradiction)]

        # Train
        for ep in range(args["trainer"]["max_epoch"]):
            next_epoch(ep, run, model, optim,
                       args["trainer"]["step_per_test"], test_env,
                       f"{log_dir}/run_{run}_log.txt")

            for step in range(args["trainer"]["step_per_epoch"]):
                obs, obs_next, action = next(training_data)
                loss = calculate_loss(model, obs, obs_next, action)

                optim.zero_grad()
                loss.backward()
                optim.step()
        
        next_epoch(ep, run, model, optim,
                   args["trainer"]["step_per_test"], test_env,
                   f"{log_dir}/run_{run}_log.txt")

        # Save model
        save_run(run, None, model, optim)

        # Test
        test_env.reset(seed=args["test_test_seed"])
        test_model(model, args["test_steps"], test_env, f"{log_dir}/test.txt")

        run += 1
        trans_path = Path(f"{args['train_dir']}/run_{run}_transitions.txt")
