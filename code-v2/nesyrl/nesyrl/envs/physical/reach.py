from typing import Tuple, List, Dict, Any
from numpy.typing import NDArray

import os
import itertools
from abc import abstractmethod

import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as pbc
import pybullet_data as pbd
from gymnasium import spaces

from nesyrl.envs.physical.base import PhysicalEnvironment, PhysicalAction, PhysicalObservation
from nesyrl.envs.physical.util import *
from nesyrl.util import spaces as uspaces


class NicoReach(PhysicalEnvironment):

    metadata = {"render_modes": ["human"]}

    render_mode: str | None

    horizon: int
    _current_step: int
    _invalid_action = False
    _max_stacks = 6

    _pbc: pbc.BulletClient
    simulation_steps: int
    _static_objects: Dict[str, int]
    _goal: int
    _init_pbc_state: int
    
    _robot: int
    _eeff: int
    _eeff_name = "endeffector"
    _active_joints: List[int]
    _active_joints_lb: NDArray
    _active_joints_ub: NDArray
    _active_joints_mf: NDArray
    _active_joints_mv: NDArray
    _max_force = 500
    _max_velocity = 0.5

    _velocity_epsilon = 1e-4
    _colision_epsilon = 1e-6

    _pb_data_dir = pbd.getDataPath()
    _models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

    _static_objects_init = {
        "floor": {
            "fileName": os.path.join(_pb_data_dir, "plane.urdf"),
            "basePosition": [0, 0, 0],
            "baseOrientation": pb.getQuaternionFromEuler([0, 0, 0])
        },
        "table": { # width: 1.5, depth: 1; height: 0.625
            "fileName": os.path.join(_pb_data_dir, "table", "table.urdf"),
            "basePosition": [0.5, 0, 0],
            "baseOrientation": pb.getQuaternionFromEuler([0, 0, np.pi / 2])
        }
    }

    _robot_init = {
        "fileName": os.path.join(_models_dir, "robots", "nico", "nico_mag.urdf"),
        "basePosition": [-0.1, 0.1, 0.625],
        "baseOrientation": pb.getQuaternionFromEuler([0, 0, 0])
    }

    _goal_init = {
        "fileName": os.path.join(_models_dir, "objects", "goal_area.urdf")
    }

    _goal_color = [0.0, 1.0, 0.0, 0.25]
    _transparent = [0.0, 1.0, 0.0, 0.0]

    _table_top = 0.625
    _block_size = 0.03
    _goal_spawning_area_x = 0.25
    _goal_spawning_area_y = [-0.18, _max_stacks * _block_size]
    _goal_spawning_area_z = [_table_top + 2 * _block_size, _table_top + 7 * _block_size]
    _goal_spawning_space_yz: spaces.Space

    _goal_size = 0.01

    _subgoal_idx: int
    _subgoals: List[List[List[float]]]
    _last_eeff_pos: List[float]
    _last_eeff_x_axis: NDArray

    _steps_in_goal: int
    _achieved_treshold: int

    _home_pos: List[float]
    _home_x_axis: NDArray
    _horizontal_x_axis = np.array([0, 0, -1])
    _target_x_axis: NDArray

    def __init__(
        self, horizon: int, simulation_steps: int,
        highlighting: bool = True, render_mode: str | None = "human"
    ) -> None:
        super().__init__()
        
        self.horizon = horizon
        self.simulation_steps = simulation_steps
        self.render_mode = render_mode

        self._init_pybullet()
        self._init_static_objects()
        self._init_goal(highlighting)
        self._init_robot()

        self._load_active_joints_and_eeff(highlighting)
        self._init_action_space()
        self._init_observation_space()
        self._init_goal_space()

        self._init_pbc_state = self._pbc.saveState()

        self._subgoal_idx = 0
        self._subgoals = []
        self._target_x_axis = self._horizontal_x_axis

        self._achieved_treshold = 240 // simulation_steps

    def _init_pybullet(self) -> None:
        if self.render_mode == "human":
            self._pbc = pbc.BulletClient(pb.GUI, options="--nogui --width=600 --height=600")
        else:
            self._pbc = pbc.BulletClient(pb.DIRECT)

        self._pbc.setGravity(0, 0, -9.81)
        self._pbc.setPhysicsEngineParameter(solverResidualThreshold=0.001,
                                            numSolverIterations=150,
                                            numSubSteps=20,
                                            useSplitImpulse=True,
                                            collisionFilterMode=1,
                                            constraintSolverType=pb.CONSTRAINT_SOLVER_LCP_DANTZIG,
                                            globalCFM=0.000001,
                                            contactBreakingThreshold=0.001,
                                            enableConeFriction=True)
        self._pbc.resetDebugVisualizerCamera(cameraDistance=0.3,
                                             cameraYaw=90,
                                             cameraPitch=-25,
                                             cameraTargetPosition=[0.2, 0, 0.85])
    
    def _init_static_objects(self) -> None:
        self._static_objects = {}
        
        for obj, settings in self._static_objects_init.items():
            self._static_objects[obj] = self._pbc.loadURDF(useFixedBase=True,
                                                           useMaximalCoordinates=True,
                                                           **settings)
    
    def _init_goal(self, highlighting: bool) -> None:
        self._goal = self._pbc.loadURDF(useMaximalCoordinates=True,
                                        **self._goal_init)
        
        color = self._goal_color if highlighting else self._transparent
        self._pbc.changeVisualShape(self._goal, -1, rgbaColor=color)
    
    def _init_robot(self) -> None:
        self._robot = self._pbc.loadURDF(useFixedBase=True,
                                         flags=pb.URDF_USE_SELF_COLLISION,
                                         **self._robot_init)

    def _load_active_joints_and_eeff(self, highlighting: bool) -> None:
        self._active_joints = []
        self._active_joints_lb = []
        self._active_joints_ub = []
        self._active_joints_mf = []
        self._active_joints_mv = []

        robot_joints = self._pbc.getNumJoints(self._robot)

        for id in range(robot_joints):
            joint_info = self._pbc.getJointInfo(self._robot, id)

            if self._eeff_name == joint_info[12].decode("utf-8"):
                self._eeff = id

                if not highlighting:
                    self._pbc.changeVisualShape(self._robot, self._eeff,
                                                rgbaColor=self._transparent)

            if joint_info[2] == pb.JOINT_FIXED:
                continue

            self._active_joints.append(id)
            self._active_joints_lb.append(joint_info[8])
            self._active_joints_ub.append(joint_info[9])
            self._active_joints_mf.append(self._max_force) # (joint_info[10])
            self._active_joints_mv.append(self._max_velocity) # (joint_info[11])
        
        self._active_joints_lb = np.array(self._active_joints_lb)
        self._active_joints_ub = np.array(self._active_joints_ub)
        self._active_joints_mf = np.array(self._active_joints_mf)
        self._active_joints_mv = np.array(self._active_joints_mv)

    def _init_action_space(self) -> None:
        max_joint_step  = 0.002 * self.simulation_steps * np.ones((len(self._active_joints),))

        self.action_space = spaces.Box(-max_joint_step, max_joint_step, dtype=float)

    def _init_observation_space(self) -> None:
        size = 3 + 3 + 3 + 3 + len(self._active_joints)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(size,), dtype=float)

    def _init_goal_space(self) -> None:
        self._goal_spawning_space_yz = spaces.Box(
            np.array([self._goal_spawning_area_y[0], self._goal_spawning_area_z[0]]),
            np.array([self._goal_spawning_area_y[1], self._goal_spawning_area_z[1]]),
            dtype=float,
            seed=self.np_random
        )

    def _get_observation(self) -> PhysicalObservation:
        eeff_pos = self._pbc.getLinkState(self._robot, self._eeff)[0]
        eeff_x_axis = self._get_eeff_x_axis()
        rel_target_x_axis = self._target_x_axis - eeff_x_axis
        goal_pos = self._pbc.getBasePositionAndOrientation(self._goal)[0]
        rel_goal_pos = np.array(goal_pos) - np.array(eeff_pos)
        joints_state = [self._pbc.getJointState(self._robot, id)[0]
                        for id in self._active_joints]

        observation = np.concatenate([np.array(eeff_pos),
                                      np.array(eeff_x_axis),
                                      np.array(rel_goal_pos),
                                      np.array(rel_target_x_axis),
                                      np.array(joints_state)
                                      ])

        return observation

    def _perform_action(self, action: PhysicalAction) -> None:
        joints_state = np.array([self._pbc.getJointState(self._robot, id)[0]
                                 for id in self._active_joints])
        action += joints_state
        action = np.clip(action, self._active_joints_lb, self._active_joints_ub)

        for i in range(len(self._active_joints)):
            self._pbc.setJointMotorControl2(bodyUniqueId=self._robot,
                                            jointIndex=self._active_joints[i],
                                            controlMode=pb.POSITION_CONTROL,
                                            targetPosition=action[i],
                                            force=self._active_joints_mf[i],
                                            maxVelocity=self._active_joints_mv[i],
                                            positionGain=0.7,
                                            velocityGain=0.3)

        self._current_step += 1

        for _ in range(self.simulation_steps):
            self._pbc.stepSimulation()
            self._invalid_action = (self._invalid_action
                                    or not self._is_valid_action())
    
    def _is_valid_action(self) -> bool:
        return not self._is_robot_collision()
    
    def _is_robot_collision(self) -> bool:
        cps = self._pbc.getContactPoints(self._robot)

        return any(cp[8] < -self._colision_epsilon for cp in cps)
    
    def _get_eeff_x_axis(self) -> NDArray:
        pos_eeff, orn_eeff = self._pbc.getLinkState(self._robot, self._eeff)[:2]
        rel_pos, rel_orn = [1, 0, 0], [0, 0, 0, 1]
        pos = self._pbc.multiplyTransforms(pos_eeff, orn_eeff, rel_pos, rel_orn)[0]
        
        return np.array(pos) - np.array(pos_eeff)
    
    def _generate_random_move(self) -> List[List[List[float]]]:
        subgoals = []
        orn = self._pbc.getQuaternionFromEuler([0, 0, 0])
        
        for _ in range(2):
            y, z = self._goal_spawning_space_yz.sample()
            subgoals.append([[self._goal_spawning_area_x, y, z], orn])

        subgoals.append([self._home_pos, orn])

        return subgoals
    
    def _activate_subgoal(self, idx: int) -> None:
        goal_pos, goal_orn = self._subgoals[idx]

        if idx == len(self._subgoals) - 1:
            self._target_x_axis = self._home_x_axis
        else:
            self._target_x_axis = self._horizontal_x_axis

        self._pbc.resetBasePositionAndOrientation(self._goal, goal_pos, goal_orn)
    
    def _evaluate_last_transition(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        if self._invalid_action:
            return -0.1, True, False, {"is_goal": False}
        
        reward = self._evaluate_movement_to_goal()
        reward += self._evaluate_movement_to_target_x_axis()
        reward += self._evaluate_movement_out()

        if self._is_current_subgoal_achieved():
            self._subgoal_idx += 1
            self._steps_in_goal = 0

            if self._subgoal_idx < len(self._subgoals):
                self._activate_subgoal(self._subgoal_idx)

            # reward = 0.2
        
        is_goal = self._subgoal_idx >= len(self._subgoals)
        truncated = self._current_step >= self.horizon
        reward = 1.0 if is_goal else reward
        
        return reward, is_goal, truncated, {"is_goal": is_goal}

    def _evaluate_movement_to_goal(self) -> float:
        goal_pos = self._pbc.getBasePositionAndOrientation(self._goal)[0]
        eeff_pos = self._pbc.getLinkState(self._robot, self._eeff)[0]

        dist_to_goal = length(np.array(goal_pos) - np.array(eeff_pos))
        last_dist_to_goal = length(np.array(goal_pos) - np.array(self._last_eeff_pos))
        
        return last_dist_to_goal - dist_to_goal
    
    def _evaluate_movement_to_target_x_axis(self) -> float:
        eeff_x_axis = self._get_eeff_x_axis()
        dist = eeff_x_axis @ self._target_x_axis
        # last_dist = self._last_eeff_x_axis @ self._target_x_axis

        # return (dist - last_dist) / 10
        return (dist - 1) / 100
    
    def _evaluate_movement_out(self) -> float:
        goal_pos = self._pbc.getBasePositionAndOrientation(self._goal)[0]
        eeff_pos = self._pbc.getLinkState(self._robot, self._eeff)[0]

        x_min = self._block_size
        x_max = self._goal_spawning_area_x
        y_min = self._goal_spawning_area_y[0] - self._block_size
        y_max = self._goal_spawning_area_y[1] + self._block_size
        z_min = goal_pos[2] #self._goal_spawning_area_z[0]
        z_max = self._goal_spawning_area_z[1] + self._block_size
        lims = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

        is_out = any(p < lim[0] or p > lim[1] for p, lim in zip(eeff_pos, lims))
        last_is_out = any(p < lim[0] or p > lim[1] for p, lim in zip(self._last_eeff_pos, lims))
        
        if not is_out and not last_is_out:
            return 0.0
        
        dist_out = 0
        for p, lim in zip(eeff_pos, lims):
            dist_out = max(max(p - lim[1], 0), lim[0] - p)
        
        last_dist_out = 0
        for p, lim in zip(self._last_eeff_pos, lims):
            last_dist_out = max(max(p - lim[1], 0), lim[0] - p)
        
        delta = last_dist_out - dist_out
        delta = 3 * delta #if delta < 0 else delta
        
        return delta
    
    def _is_current_subgoal_achieved(self) -> bool:
        goal_pos = self._pbc.getBasePositionAndOrientation(self._goal)[0]
        eeff_pos = self._pbc.getLinkState(self._robot, self._eeff)[0]
        
        in_goal = (squared_length(np.array(goal_pos) - np.array(eeff_pos))
                   < self._goal_size ** 2)
        self._steps_in_goal += in_goal

        return self._steps_in_goal > self._achieved_treshold
    
    def create_snapshot(self) -> Any:
        snapshot = {
            "current_step": self._current_step,
            "invalid_action": self._invalid_action,
            "pbc_state": self._pbc.saveState(),
            "subgoal_idx": self._subgoal_idx,
            "subgoals": self._subgoals,
            "last_eeff_pos": self._last_eeff_pos,
            "last_eeff_x_axis": self._last_eeff_x_axis,
            "steps_in_goal": self._steps_in_goal,
            "target_x_axis": self._target_x_axis
        }

        return snapshot
    
    def restore_snapshot(self, snapshot: Any) -> PhysicalObservation:
        self._current_step = snapshot["current_step"]
        self._invalid_action = snapshot["invalid_action"]
        self._subgoal_idx = snapshot["subgoal_idx"]
        self._subgoals = snapshot["subgoals"]
        self._last_eeff_pos = snapshot["last_eeff_pos"]
        self._last_eeff_x_axis = snapshot["last_eeff_x_axis"]
        self._steps_in_goal = snapshot["steps_in_goal"]
        self._target_x_axis = snapshot["target_x_axis"]

        self._pbc.restoreState(snapshot["pbc_state"])

        return self._get_observation()
    
    def step(self, action: PhysicalAction) -> Tuple[PhysicalObservation, float, bool, bool, Dict[str, Any]]:
        self._perform_action(action)
        reward, terminated, truncated, info = self._evaluate_last_transition()

        self._last_eeff_pos = self._pbc.getLinkState(self._robot, self._eeff)[0]
        self._last_eeff_x_axis = self._get_eeff_x_axis()

        observation = self._get_observation()

        return observation, reward, terminated, truncated, info
    
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[PhysicalObservation, Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self._current_step = 0
        self._invalid_action = False
        self._subgoal_idx = 0
        self._steps_in_goal = 0
        
        self._pbc.restoreState(self._init_pbc_state)

        self._home_pos = self._pbc.getLinkState(self._robot, self._eeff)[0]
        self._home_x_axis = self._get_eeff_x_axis()
        self._subgoals = self._generate_random_move()
        self._activate_subgoal(self._subgoal_idx)

        self._last_eeff_pos = self._pbc.getLinkState(self._robot, self._eeff)[0]
        self._last_eeff_x_axis = self._get_eeff_x_axis()

        observation = self._get_observation()

        return observation, {}
        
    def close(self) -> None:
        self._pbc.disconnect()
