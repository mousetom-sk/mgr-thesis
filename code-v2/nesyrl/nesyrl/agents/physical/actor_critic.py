from typing import List, Tuple, Dict, Any

from numpy.typing import NDArray

import torch
from torch import Tensor
from torch.nn import Module

from tianshou.utils.net.continuous import ActorProb


class DualActorProb(Module):

    actor0: ActorProb
    actor1: ActorProb

    def __init__(self, actor0: ActorProb, actor1: ActorProb) -> None:
        super().__init__()

        self.actor0 = actor0
        self.actor1 = actor1
        
        for i, a in enumerate((actor0, actor1)):
            self.add_module(f"actor{i}", a)

    def forward(
        self, obs: NDArray | Tensor,
        state: Any = None, info: Dict[str, Any] = {}
    ) -> Tuple[Tensor, Any]:
        mu0, sigma0 = self.actor0(obs[:, :-1])[0]
        mu1, sigma1 = self.actor1(obs[:, :-1])[0]
        
        obs = torch.tensor(obs).to(mu0)
        mask = obs[:, -1] > 0
        mu = torch.where(mask, mu1, mu0)
        sigma = torch.where(mask, sigma1, sigma0)

        return (mu, sigma), state
