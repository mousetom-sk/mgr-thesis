from typing import Callable, Optional, Tuple

from tianshou.utils.logger.base import LOG_DATA_TYPE, BaseLogger


class FileLogger(BaseLogger):
    
    def __init__(self, path: str) -> None:
        super().__init__(0, 1, 0)
        
        self.path = path

    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        if step_type == "test":
            record = [data["reward"], data["reward_std"],
                      data["length"], data["length_std"],
                      data["goal"], data["goal_std"]]
            
            with open(self.path, "a") as out:
                print(", ".join(str(d) for d in record), file=out)

    def log_test_data(self, collect_result: dict, step: int) -> None:
        log_data = {
            "reward": collect_result["rew"],
            "reward_std": collect_result["rew_std"],
            "length": collect_result["len"],
            "length_std": collect_result["len_std"],
            "goal": collect_result["goal"],
            "goal_std": collect_result["goal_std"],
        }

        self.write("test", step, log_data)
    
    def log_train_data(self, collect_result: dict, step: int) -> None:
        return

    def log_update_data(self, update_result: dict, step: int) -> None:
        return

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
    ) -> None:
        return

    def restore_data(self) -> Tuple[int, int, int]:
        return 0, 0, 0
