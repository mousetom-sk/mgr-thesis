from typing import Any, Dict, Optional

from tianshou.trainer import OnpolicyTrainer as OnpolicyBase

class OnpolicyTrainer(OnpolicyBase):

    def policy_update_fn(
        self, data: Dict[str, Any], result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Perform one on-policy update."""
        assert self.train_collector is not None
        losses = self.policy.update(
            0,
            self.train_collector.buffer,
            batch_size=self.batch_size,
            repeat=self.repeat_per_collect,
        )
        step = max([1] + [len(v) for v in losses.values() if isinstance(v, list)])
        self.gradient_step += step
        self.log_update_data(data, losses)
