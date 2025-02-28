from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn.functional as F

from tianshou.data import Batch, to_torch, to_torch_as
from tianshou.policy import (PGPolicy as PGBase,
                             A2CPolicy as A2CBase,
                             PPOPolicy as PPOBase)

from nesyrl.agents.symbolic.actor_critic import NLActor, NLCritic


class PGPolicy(PGBase):

    model: NLActor
    
    def __init__(
        self,
        model: NLActor,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        deterministic_eval: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model, optim, dist_fn, discount_factor, reward_normalization,
            action_scaling, action_bound_method, deterministic_eval,
            **kwargs
        )

    def learn(
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses = []
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                self.optim.zero_grad()
                result = self(minibatch)
                dist = result.dist
                act = to_torch_as(minibatch.act, result.act)
                ret = to_torch(minibatch.returns, torch.float, result.act.device)
                log_prob = dist.log_prob(act).reshape(len(ret), -1).transpose(0, 1)
                loss = -(log_prob * ret).mean()
                loss += self.model.compute_regularization_loss()
                loss.backward()
                self.optim.step()
                self.model.postprocess_parameters()
                losses.append(loss.item())

        return {"loss": losses}


class A2CPolicy(A2CBase):

    actor: NLActor
    critic: NLCritic

    def __init__(
        self,
        actor: NLActor,
        critic: NLCritic,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: Optional[float] = None,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        **kwargs: Any
    ) -> None:
        super().__init__(
            actor, critic, optim, dist_fn,
            vf_coef, ent_coef, max_grad_norm, gae_lambda, max_batchsize,
            **kwargs
        )
    
    def learn(
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, actor_losses, vf_losses, ent_losses = [], [], [], []
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                # calculate loss for actor
                dist = self(minibatch).dist
                log_prob = dist.log_prob(minibatch.act)
                log_prob = log_prob.reshape(len(minibatch.adv), -1).transpose(0, 1)
                actor_loss = -(log_prob * minibatch.adv).mean()
                # calculate loss for critic
                value = self.critic(minibatch.obs).flatten()
                vf_loss = F.mse_loss(minibatch.returns, value)
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = actor_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * ent_loss
                loss += self.actor.compute_regularization_loss()
                loss += self.critic.compute_regularization_loss()
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    torch.nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()
                self.actor.postprocess_parameters()
                self.critic.postprocess_parameters()
                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return {
            "loss": losses,
            "loss/actor": actor_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }


class PPOPolicy(PPOBase):
    
    actor: NLActor
    critic: NLCritic

    def __init__(
        self,
        actor: NLActor,
        critic: NLCritic,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        eps_clip: float = 0.2,
        dual_clip: Optional[float] = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            actor, critic, optim, dist_fn,
            eps_clip, dual_clip, value_clip,
            advantage_normalization, recompute_advantage,
            **kwargs
        )

    def learn(
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        for step in range(repeat):
            if self._recompute_adv and step > 0:
                batch = self._compute_returns(batch, self._buffer, self._indices)
            for minibatch in batch.split(batch_size, merge_last=True):
                # calculate loss for actor
                dist = self(minibatch).dist
                if self._norm_adv:
                    mean, std = minibatch.adv.mean(), minibatch.adv.std()
                    minibatch.adv = (minibatch.adv -
                                     mean) / (std + self._eps)  # per-batch norm
                ratio = (dist.log_prob(minibatch.act) -
                         minibatch.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * minibatch.adv
                surr2 = ratio.clamp(
                    1.0 - self._eps_clip, 1.0 + self._eps_clip
                ) * minibatch.adv
                if self._dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self._dual_clip * minibatch.adv)
                    clip_loss = -torch.where(minibatch.adv < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                # calculate loss for critic
                value = self.critic(minibatch.obs).flatten()
                if self._value_clip:
                    v_clip = minibatch.v_s + \
                        (value - minibatch.v_s).clamp(-self._eps_clip, self._eps_clip)
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                # calculate regularization and overall loss
                ent_loss = dist.entropy().mean()
                loss = clip_loss + self._weight_vf * vf_loss \
                    - self._weight_ent * ent_loss
                loss += self.actor.compute_regularization_loss()
                loss += self.critic.compute_regularization_loss()
                self.optim.zero_grad()
                loss.backward()
                if self._grad_norm:  # clip large gradient
                    torch.nn.utils.clip_grad_norm_(
                        self._actor_critic.parameters(), max_norm=self._grad_norm
                    )
                self.optim.step()
                self.actor.postprocess_parameters()
                self.critic.postprocess_parameters()
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return {
            "loss": losses,
            "loss/clip": clip_losses,
            "loss/vf": vf_losses,
            "loss/ent": ent_losses,
        }
