from typing import Union

import numpy as np

from tianshou.env.venvs import BaseVectorEnv
from tianshou.env.venv_wrappers import VectorEnvNormObs as VEnvNormObsBase
from tianshou.utils import RunningMeanStd as RMSBase


class RunningMeanStd(RMSBase):

    def norm(self, data_array: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        normalized = super().norm(data_array)
        
        if isinstance(data_array, float):
            return normalized
        
        normalized[:, -1] = data_array[:, -1]
        
        return normalized


class VectorEnvNormObs(VEnvNormObsBase):

    def __init__(self, venv: BaseVectorEnv, update_obs_rms: bool = True):
        super().__init__(venv, update_obs_rms)

        self.obs_rms = RunningMeanStd()

    def set_obs_rms(self, obs_rms: RunningMeanStd) -> None:
        return super().set_obs_rms(obs_rms)
    
    def get_obs_rms(self) -> RunningMeanStd:
        return super().get_obs_rms()
