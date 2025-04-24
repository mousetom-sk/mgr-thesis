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


class NicoBlocksWorldBase(PhysicalEnvironment):

    metadata = {"render_modes": ["human"]}

    eeff_obs_key = "eeff"
    goal_obs_key = "goal"
    render_mode: str | None

    _table = "table"

    _horizon: int
    _current_step: int
    _current_blocks_state: List[List[str]]
    _stacking_base: List[int]
    _invalid_action = False
    _max_stacks = 6

    _pbc: pbc.BulletClient
    _simulation_steps: int
    _static_objects: Dict[str, int]
    _blocks: Dict[str, int]
    _goal: int
    _init_pbc_state: int
    
    _robot: int
    _eeff: int
    _eeff_name = "endeffector"
    _eeff_center_name: int
    _eeff_center_name = "endeffector_center"
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
        _table: { # width: 1.5, depth: 1; height: 0.625
            "fileName": os.path.join(_pb_data_dir, "table", "table.urdf"),
            "basePosition": [0.5, 0, 0],
            "baseOrientation": pb.getQuaternionFromEuler([0, 0, np.pi / 2])
        }
    }

    _block_init = {
        "fileName": os.path.join(_models_dir, "objects", "cube.urdf"),
        "globalScaling": 0.6
    }
    
    _block_colors = {
        "a": [1.0, 0.0, 0.0, 1.0],
        "b": [0.0, 1.0, 0.0, 1.0],
        "c": [0.0, 0.0, 1.0, 1.0],
        "d": [1.0, 0.8, 0.0, 1.0],
        "e": [0.5, 0.0, 1.0, 1.0],
        "f": [1.0, 0.5, 0.0, 1.0],
        "g": [1.0, 1.0, 1.0, 1.0],
        "h": [1.0, 1.0, 1.0, 1.0],
        "i": [1.0, 1.0, 1.0, 1.0],
        "j": [1.0, 1.0, 1.0, 1.0],
        "k": [1.0, 1.0, 1.0, 1.0],
        "l": [1.0, 1.0, 1.0, 1.0],
        "m": [1.0, 1.0, 1.0, 1.0],
        "n": [1.0, 1.0, 1.0, 1.0],
    }

    _robot_init = {
        "fileName": os.path.join(_models_dir, "robots", "nico", "nico_mag.urdf"),
        "basePosition": [-0.1, 0.1, 0.625],
        "baseOrientation": pb.getQuaternionFromEuler([0, 0, 0])
    }

    _goal_init = {
        "fileName": os.path.join(_models_dir, "objects", "goal_area.urdf"),
        # "globalScaling": 0.6
    }

    _goal_color = [0.0, 1.0, 0.0, 0.25]
    _transparent = [0.0, 1.0, 0.0, 0.0]

    _table_top = 0.625
    _block_size = 0.05 * _block_init["globalScaling"]
    _blocks_x = 0.25
    _blocks_y_start = -0.18
    _blocks_y_delta = 5 * _block_size / 2
    _blocks_z_start = _table_top + _block_size / 2
    _blocks_z_delta = _block_size

    _goal_size = 0.01 #0.05 * _goal_init["globalScaling"]
    _goal_shift = [0, 0, 3 * _block_size / 2]

    def __init__(
        self, horizon: int, blocks: List[str], simulation_steps: int,
        highlighting: bool = True, render_mode: str | None = "human"
    ) -> None:
        super().__init__()

        unknown_blocks = set(blocks) - set(self._block_colors)

        if len(unknown_blocks) > 0:
            raise ValueError(f"Unknown blocks: {unknown_blocks}")
        
        self._horizon = horizon
        self._simulation_steps = simulation_steps
        self.render_mode = render_mode

        self._init_pybullet()
        self._init_static_objects()
        self._init_blocks(blocks)
        self._init_goal(highlighting)
        self._init_robot()

        self._stacking_base = [0] * len(blocks) + [1] * (self._max_stacks - 1)

        self._load_active_joints_and_eeff(highlighting)
        self._init_action_space()
        self._init_observation_space()

        self._init_pbc_state = self._pbc.saveState()

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
    
    def _init_blocks(self, blocks: List[str]) -> None:
        self._blocks = {}
        
        for b in blocks:
            self._blocks[b] = self._pbc.loadURDF(useMaximalCoordinates=True,
                                                 **self._block_init)
            self._pbc.changeVisualShape(self._blocks[b], -1,
                                        rgbaColor=self._block_colors[b])
    
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
            elif self._eeff_center_name == joint_info[12].decode("utf-8"):
                self._eeff_center = id

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
        max_joint_step  = 0.002 * self._simulation_steps * np.ones((len(self._active_joints),))

        self.action_space = spaces.Box(-max_joint_step, max_joint_step, dtype=float)

    def _init_observation_space(self) -> None:
        self.observation_space = spaces.Dict(
            {b: spaces.Box(-np.inf, np.inf, shape=(7,), dtype=float) for b in self._blocks}
            | {f"joint_{id}": uspaces.Float(-np.inf, np.inf) for id in self._active_joints}
            | {self.eeff_obs_key: spaces.Box(-np.inf, np.inf, shape=(7,), dtype=float)}
            | {self.goal_obs_key: spaces.Box(-np.inf, np.inf, shape=(7,), dtype=float)}
        )

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

        for _ in range(self._simulation_steps):
            self._pbc.stepSimulation()
            self._invalid_action = (self._invalid_action
                                    or not self._is_valid_action())
    
    def _get_observation(self) -> PhysicalObservation:
        observation = {}
        
        pos_eeff, orn_eeff = self._pbc.getLinkState(self._robot, self._eeff)[:2]
        inv_pos_eeff, inv_orn_eeff = self._pbc.invertTransform(pos_eeff, orn_eeff)

        observation[self.eeff_obs_key] = np.concatenate(
            [np.array(v) for v in (pos_eeff, orn_eeff)]
        )

        for b in self._blocks:
            pos, orn = self._pbc.getBasePositionAndOrientation(self._blocks[b])
            block_in_eeff = self._pbc.multiplyTransforms(inv_pos_eeff, inv_orn_eeff, pos, orn)
            observation[b] = np.concatenate([np.array(v) for v in block_in_eeff])
        
        for id in self._active_joints:
            joint_state = self._pbc.getJointState(self._robot, id)
            observation[f"joint_{id}"] = joint_state[0]

        return observation

    def _is_robot_collision(self) -> bool:
        cps = self._pbc.getContactPoints(self._robot)

        return any(cp[8] < -self._colision_epsilon for cp in cps)
    
    def _is_valid_action(self) -> bool:
        return not self._is_robot_collision()

    def _generate_random_blocks_state(self) -> List[List[str]]:
        ordering = iter(self.np_random.permutation(list(self._blocks.keys())))
        stacking = self.np_random.permutation(self._stacking_base)
        stacking = np.concatenate((stacking, [1]))

        state = []
        stack = []

        for end in stacking:
            if end:
                state.append(stack)
                stack = []
            else:
                stack.append(next(ordering))

        return state
    
    def _set_blocks_state(self, state: List[List[str]]) -> None:
        orn = [0, 0, 0, 1]
        bottom_x = self._blocks_x
        bottom_y = self._blocks_y_start
        bottom_z = self._blocks_z_start

        for stack in state:
            pos = np.array([bottom_x, bottom_y, bottom_z])

            for i, b in enumerate(stack):
                z_offset = i * np.array([0, 0, self._blocks_z_delta])
                z_offset[2] += self._colision_epsilon

                self._pbc.resetBasePositionAndOrientation(self._blocks[b],
                                                          pos + z_offset,
                                                          orn)
                self._pbc.resetBaseVelocity(self._blocks[b], [0, 0, 0], [0, 0, 0])
                self._pbc.stepSimulation()

            bottom_y += self._blocks_y_delta
    
    @abstractmethod
    def _evaluate_last_transition(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        pass

    def create_snapshot(self) -> Any:
        snapshot = {
            "current_step": self._current_step,
            "current_blocks_state": self._current_blocks_state,
            "invalid_action": self._invalid_action,
            "pbc_state": self._pbc.saveState()
        }

        return snapshot
    
    def restore_snapshot(self, snapshot: Any) -> PhysicalObservation:
        self._current_step = snapshot["current_step"]
        self._current_blocks_state = snapshot["current_blocks_state"]
        self._invalid_action = snapshot["invalid_action"]

        self._pbc.restoreState(snapshot["pbc_state"])

        return self._get_observation()
    
    def step(self, action: PhysicalAction) -> Tuple[PhysicalObservation, float, bool, bool, Dict[str, Any]]:
        self._perform_action(action)
        observation = self._get_observation()
        reward, terminated, truncated, info = self._evaluate_last_transition()
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[PhysicalObservation, Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self._current_step = 0
        self._invalid_action = False
        
        self._pbc.restoreState(self._init_pbc_state)

        self._current_blocks_state = self._generate_random_blocks_state()
        self._set_blocks_state(self._current_blocks_state)

        observation = self._get_observation()

        return observation, {}
        
    def close(self) -> None:
        self._pbc.disconnect()


class NicoBlocksWorldMove(NicoBlocksWorldBase):

    _home = "home"
    _middle = "middle"
    _subgoal_idx: int
    _subgoals: List[str]
    _last_dist_to_goal: float
    _last_collider_pos: List[float]
    _last_dist_to_closest: float
    _last_eeff_x_axis: NDArray

    _steps_in_goal: int
    _achieved_treshold: int

    _magnetized_block: str
    _magnetization_constraint: int

    _home_pos: List[float]
    _home_x_axis: NDArray
    _horizontal_x_axis = np.array([0, 0, -1])
    _target_x_axis: NDArray

    def __init__(
        self, horizon: int, blocks: List[str], simulation_steps: int,
        highlighting: bool = True, render_mode: str | None = "human"
    ) -> None:
        super().__init__(horizon, blocks, simulation_steps, highlighting, render_mode)

        self._subgoal_idx = 0
        self._subgoals = []
        self._magnetized_block = None
        self._target_x_axis = self._horizontal_x_axis

        self._achieved_treshold = 240 // simulation_steps

    def _init_observation_space(self) -> None:
        size = 3 + 3 + 3 + len(self._active_joints) + 3 + 3 #+ 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(size,), dtype=float)

    def _get_observation(self) -> PhysicalObservation:
        pos_eeff = self._pbc.getLinkState(self._robot, self._eeff)[0]
        x_axis = self._get_eeff_x_axis()
        rel_target_x_axis = self._target_x_axis - x_axis
        pos_goal = self._pbc.getBasePositionAndOrientation(self._goal)[0]
        rel_pos_goal = np.array(pos_goal) - np.array(pos_eeff)
        pos_closest = self._get_closest_block_info()[1]
        rel_pos_closest = np.array(pos_closest) - np.array(pos_eeff)
        if self._magnetized_block is not None:
            rel_len = length(rel_pos_closest)
            rel_pos_closest = rel_pos_closest * (rel_len - self._block_size) / rel_len
        joints_state = [self._pbc.getJointState(self._robot, id)[0]
                        for id in self._active_joints]

        observation = np.concatenate([np.array(pos_eeff),
                                      np.array(x_axis),
                                      np.array(rel_pos_goal),
                                      np.array(rel_target_x_axis),
                                      np.array(rel_pos_closest),
                                      np.array(joints_state),
                                    #   np.array([self._magnetized_block is not None])
                                      ])

        return observation
    
    def _get_closest_block_info(self) -> Tuple[str, List[int], float]:
        pos_collider, orn_collider = self._get_collider_pose()
        inv_pos_collider, inv_orn_collider = self._pbc.invertTransform(pos_collider, orn_collider)

        block_postions = []

        for b in self._blocks:
            if self._magnetized_block is not None and b == self._magnetized_block:
                continue

            # if self._subgoal_idx == 0 and len(self._subgoals) > 0 and b == self._subgoals[0]:
            #     continue

            pos, orn = self._pbc.getBasePositionAndOrientation(self._blocks[b])
            block_in_collider = self._pbc.multiplyTransforms(inv_pos_collider, inv_orn_collider, pos, orn)
            block_postions.append((b, list(pos), length(block_in_collider[0])))

        closest_info = sorted(block_postions, key=lambda x: x[2])[0]

        return closest_info
    
    def _generate_random_move(self) -> List[str]:
        subgoals = []
        valid_moves = []
        # table_to_table = []

        for i, stack1 in enumerate(self._current_blocks_state):
            if len(stack1) == 0:
                continue
            
            block = stack1[-1]
            on_table = len(stack1) == 1

            for j, stack2 in enumerate(self._current_blocks_state):
                if j == i or (on_table and len(stack2) == 0
                               and all(len(self._current_blocks_state[k]) == 0
                                       for k in range(min(i, j) + 1, max(i, j)))):
                    continue

                # if j == i or (on_table and len(stack2) == 0):
                #     continue

                # if on_table and len(stack2) == 0:
                #     table_to_table.append((block, f"{self._table}{j}"))
                # el
                if len(stack2) == 0:
                    valid_moves.append((block, f"{self._table}{j}"))
                    # valid_moves.append((block, f"{self._middle}{self._table}{j}", f"{self._table}{j}"))
                    # valid_moves.append((block, self._home, f"{self._table}{j}"))
                else:
                    valid_moves.append((block, stack2[-1]))
                    # valid_moves.append((block, f"{self._middle}{stack2[-1]}", stack2[-1]))
                    # valid_moves.append((block, self._home, stack2[-1]))
        
        # if len(table_to_table) > 0 and self.np_random.random() < 0.6:
        #     choices = table_to_table
        # else:
        # choices = valid_moves
        
        subgoals.extend(self.np_random.choice(valid_moves))
        subgoals.append(self._home)

        return subgoals
    
    def _get_subgoal_pose(self, subgoal: str, extra_shift: bool = False) -> Tuple[List[float], List[float]]:
        if subgoal in self._blocks:
            pos, orn = self._pbc.getBasePositionAndOrientation(self._blocks[subgoal])
            rel_pos, rel_orn = list(self._goal_shift), [0, 0, 0, 1]

            if extra_shift:
                rel_pos[2] += self._blocks_z_delta

            goal_pos, goal_orn = self._pbc.multiplyTransforms(pos, orn, rel_pos, rel_orn)
        elif subgoal.startswith(self._table):
            i = int(subgoal[len(self._table):])

            pos = [self._blocks_x,
                   self._blocks_y_start + i * self._blocks_y_delta,
                   self._blocks_z_start]
            orn = [0, 0, 0, 1]
            rel_pos, rel_orn = list(self._goal_shift), [0, 0, 0, 1]

            goal_pos, goal_orn = self._pbc.multiplyTransforms(pos, orn, rel_pos, rel_orn)
        elif subgoal.startswith(self._middle):
            target = subgoal[len(self._middle):]
            target_pos = self._get_subgoal_pose(target, extra_shift)[0]
            eeff_pos = self._pbc.getLinkState(self._robot, self._eeff)[0]
            middle_pos = (np.array(target_pos) + np.array(eeff_pos)) / 2
            shift = length(middle_pos - np.array(eeff_pos))

            # goal_pos = [middle_pos[0] - shift, middle_pos[1], middle_pos[2]]
            goal_pos = [max(middle_pos[0] - shift, 0.1), middle_pos[1], middle_pos[2]]
            goal_orn = [0, 0, 0, 1]
        else:
            goal_pos, goal_orn = self._home_pos, [0, 0, 0, 1]

        return goal_pos, goal_orn
    
    def _activate_subgoal(self, subgoal: str, extra_shift: bool = False) -> None:
        goal_pos, goal_orn = self._get_subgoal_pose(subgoal, extra_shift)

        if subgoal == self._home:
            self._target_x_axis = self._home_x_axis
        else:
            self._target_x_axis = self._horizontal_x_axis

        self._pbc.resetBasePositionAndOrientation(self._goal, goal_pos, goal_orn)

    def _get_distance_to_goal(self) -> float:
        pos_eeff, orn_eeff = self._pbc.getLinkState(self._robot, self._eeff)[:2]
        inv_pos_eeff, inv_orn_eeff = self._pbc.invertTransform(pos_eeff, orn_eeff)

        pos_goal, orn_goal = self._pbc.getBasePositionAndOrientation(self._goal)
        goal_in_eeff = self._pbc.multiplyTransforms(inv_pos_eeff, inv_orn_eeff,
                                                    pos_goal, orn_goal)
        
        return length(goal_in_eeff[0])

    def _get_collider_pose(self) -> Tuple[List[float], List[float]]:
        # if self._magnetized_block is None:
        #     pos, orn = self._pbc.getLinkState(self._robot, self._eeff_center)[:2]
        # else:
        #     pos1, orn = self._pbc.getLinkState(self._robot, self._eeff_center)[:2]
        #     pos2 = self._pbc.getLinkState(self._robot, self._eeff)[0]

        #     pos = (np.array(pos1) + np.array(pos2)) / 2

        pos, orn = self._pbc.getLinkState(self._robot, self._eeff)[:2]

        return list(pos), list(orn)

    def _get_eeff_x_axis(self) -> NDArray:
        pos_eeff, orn_eeff = self._pbc.getLinkState(self._robot, self._eeff)[:2]
        rel_pos, rel_orn = [1, 0, 0], [0, 0, 0, 1]
        pos = self._pbc.multiplyTransforms(pos_eeff, orn_eeff, rel_pos, rel_orn)[0]
        
        return np.array(pos) - np.array(pos_eeff)

    def _is_current_subgoal_achieved(self) -> bool:
        pos_goal = self._pbc.getBasePositionAndOrientation(self._goal)[0]
        pos_eeff = self._pbc.getLinkState(self._robot, self._eeff)[0]

        # in_goal =  all(pos_goal[i] - self._goal_size / 2
        #                < pos_eeff[i] <
        #                pos_goal[i] + self._goal_size / 2
        #                for i in range(3))
        
        in_goal = squared_length(np.array(pos_goal) - np.array(pos_eeff)) < self._goal_size ** 2
        self._steps_in_goal += in_goal

        return self._steps_in_goal > self._achieved_treshold
    
    def _magnetize_block(self, block: str) -> None:
        # x_axis = self._get_eeff_x_axis()

        # if abs(x_axis @ self._horizontal_x_axis) < 0.94:
        #     self._invalid_action = True
        #     return
        
        self._magnetized_block = block
        pos_eeff, orn_eeff = self._pbc.getLinkState(self._robot, self._eeff)[:2]
        rel_pos, rel_orn = [self._block_size / 2, 0, 0], [0, 0, 0, 1]
        pos, orn = self._pbc.multiplyTransforms(pos_eeff, orn_eeff, rel_pos, rel_orn)

        self._pbc.resetBasePositionAndOrientation(self._blocks[block], pos, orn)
        self._magnetization_constraint = self._pbc.createConstraint(
            self._robot, self._eeff, self._blocks[block], -1, pb.JOINT_FIXED,
            [0, 0, 0], rel_pos, [0, 0, 0], rel_orn, [0, 0, 0, 1]
        )

    def _demagnetize_block(self, to: str) -> None:
        # x_axis = self._get_eeff_x_axis()

        # if abs(x_axis @ self._horizontal_x_axis) < 0.94:
        #     self._invalid_action = True
        #     return
        
        orn = [0, 0, 0, 1]
        
        if to in self._blocks:
            pos, orn = self._pbc.getBasePositionAndOrientation(self._blocks[to])
            rel_pos, rel_orn = [0, 0, self._blocks_z_delta + self._colision_epsilon], [0, 0, 0, 1]

            pos, orn = self._pbc.multiplyTransforms(pos, orn, rel_pos, rel_orn)
        else:
            i = int(to[len(self._table):])

            pos = [self._blocks_x,
                   self._blocks_y_start + i * self._blocks_y_delta,
                   self._blocks_z_start + self._colision_epsilon]

        self._pbc.removeConstraint(self._magnetization_constraint)
        self._pbc.resetBasePositionAndOrientation(self._blocks[self._magnetized_block], pos, orn)
        self._pbc.resetBaseVelocity(self._blocks[self._magnetized_block], [0, 0, 0], [0, 0, 0])
        self._magnetized_block = None

        self._pbc.stepSimulation()

    def _is_magnetized_block_collision(self) -> bool:
        if self._magnetized_block is None:
            return False
        
        cps = self._pbc.getContactPoints(self._blocks[self._magnetized_block])

        return any(cp[8] < -self._colision_epsilon for cp in cps)

    def _is_moving_block(self) -> bool:
        velocities = list(self._pbc.getBaseVelocity(self._blocks[b])[0]
                          for b in self._blocks
                          if self._magnetized_block is None
                          or b != self._magnetized_block)

        return any(squared_length(v) >= (self._velocity_epsilon ** 2)
                   for v in velocities)
    
    def _is_valid_action(self) -> bool:
        return (super()._is_valid_action()
                and not self._is_moving_block()
                and not self._is_magnetized_block_collision())

    def _evaluate_movement_to_goal(self) -> float:
        dist_to_goal = self._get_distance_to_goal()
        
        return self._last_dist_to_goal - dist_to_goal
        
        # return - dist_to_goal
    
    def _evaluate_movement_to_closest(self) -> float:
        closest, _, dist_to_closest = self._get_closest_block_info()
        boundary = 3 * self._block_size / 2

        if self._magnetized_block is None:
            # boundary = 3 * self._block_size / 2
            last_dist_to_closest = self._last_dist_to_closest
        else:
            # boundary = 7 * self._block_size / 3
            dist_to_closest -= self._block_size
            last_dist_to_closest = self._last_dist_to_closest - self._block_size

        # delta = -0.01 if dist_to_closest < boundary else 0.0

        if min(dist_to_closest, last_dist_to_closest) > boundary:
            return False, 0.0
        
        delta = min(dist_to_closest, boundary) - min(last_dist_to_closest, boundary)
        delta = 3 * delta if delta < 0 else delta


        # if self._magnetized_block is not None:
        #     dist_to_closest -= self._block_size

        # if dist_to_closest > boundary:
        #     return False, 0.0
        
        # delta = dist_to_closest - boundary


        # if delta < 0:
        #     delta = 3 * delta
        # else:
        #     collider_pos = self._get_collider_pose()[0]
        #     dist_behind_blocks = collider_pos[0] - self._blocks_x
        #     last_dist_behind_blocks = self._last_collider_pos[0] - self._blocks_x

        #     if min(dist_behind_blocks, last_dist_behind_blocks) > 0:
        #         delta = 0.0
        #     else:
        #         delta = last_dist_behind_blocks - dist_behind_blocks
        #         delta = max(delta, 0.0)

        # if self._magnetized_block is None:
        #     delta /= 4

        return True, delta

        # if dist_to_closest > 2 * self._block_size:
        #     return 0.0
        
        # return dist_to_closest - 2 * self._block_size
    
    def _evaluate_movement_to_table(self) -> float:
        collider_pos = self._get_collider_pose()[0]
        dist_to_table = collider_pos[2] - self._table_top
        last_dist_to_table = self._last_collider_pos[2] - self._table_top
        boundary = 2 * self._block_size

        # delta = -0.01 if dist_to_table < boundary else 0.0

        if min(dist_to_table, last_dist_to_table) > boundary:
            return False, 0.0
        
        delta = min(dist_to_table, boundary) - min(last_dist_to_table, boundary)
        delta = 3 * delta if delta < 0 else delta


        # if dist_to_table > boundary:
        #     return False, 0.0
        
        # delta = dist_to_table - boundary


        # if self._magnetized_block is None:
        #     delta /= 4
        
        return True, delta

        # if dist_to_table > self._block_size:
        #     return 0.0
        
        # return dist_to_table - self._block_size
    
    def _evaluate_movement_behind_blocks(self) -> float:
        collider_pos = self._get_collider_pose()[0]
        dist_behind_blocks = collider_pos[0] - self._blocks_x
        last_dist_behind_blocks = self._last_collider_pos[0] - self._blocks_x

        goal_pos = self._pbc.getBasePositionAndOrientation(self._goal)[0]

        if abs(goal_pos[1] - collider_pos[1]) < 3 * self._block_size / 2:
            boundary = 0
        else:
            boundary = -self._block_size
        # boundary = 0
        # delta = -0.01 if dist_behind_blocks < boundary else 0.0

        if max(dist_behind_blocks, last_dist_behind_blocks) < boundary:
            return False, 0.0
        
        delta = max(last_dist_behind_blocks, boundary) - max(dist_behind_blocks, boundary)
        delta = 3 * delta if delta < 0 else delta
        # delta = 3 * delta if delta < 0 and dist_behind_blocks > 0 else delta

        
        # if dist_behind_blocks < boundary:
        #     return False, 0.0
        
        # delta = boundary - dist_behind_blocks


        return True, delta

        # if dist_behind_blocks < boundary:
        #     return 0.0
        
        # return boundary - dist_behind_blocks
    
    def _evaluate_movement_to_target_x_axis(self) -> float:
        x_axis = self._get_eeff_x_axis()
        dist = x_axis @ self._target_x_axis
        last_dist = self._last_eeff_x_axis @ self._target_x_axis

        return (dist - last_dist) / 10
        # return (dist - 1) / 100
    
    def _evaluate_last_transition(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        if self._invalid_action:
            return -0.1, True, False, {"is_goal": False}
        
        active, reward = self._evaluate_movement_to_closest()
        a, r = self._evaluate_movement_to_table()
        active, reward = active or a, reward + r
        a, r = self._evaluate_movement_behind_blocks()
        active, reward = active or a, reward + r
        reward += self._evaluate_movement_to_target_x_axis()
        reward += self._evaluate_movement_to_goal()
            
        # reward /= 100

        if self._is_current_subgoal_achieved():
            self._subgoal_idx += 1
            self._steps_in_goal = 0

            if self._subgoal_idx < len(self._subgoals):
                self._activate_subgoal(self._subgoals[self._subgoal_idx], True)
            
            if self._subgoal_idx == 1:
                self._magnetize_block(self._subgoals[self._subgoal_idx - 1])

            if self._subgoal_idx == len(self._subgoals) - 1:
                self._demagnetize_block(self._subgoals[self._subgoal_idx - 1])

            # reward = 0.2
        
        is_goal = self._subgoal_idx >= len(self._subgoals)
        truncated = self._current_step >= self._horizon
        reward = 1.0 if is_goal else reward
        
        return reward, is_goal, truncated, {"is_goal": is_goal}

    def create_snapshot(self) -> Any:
        snapshot = {
            "subgoal_idx": self._subgoal_idx,
            "subgoals": self._subgoals,
            "last_dist_to_goal": self._last_dist_to_goal,
            "last_collider_pos": self._last_collider_pos,
            "last_dist_to_closest": self._last_dist_to_closest,
            "last_eeff_x_axis": self._last_eeff_x_axis,
            "steps_in_goal": self._steps_in_goal,
            "magnetized_block": self._magnetized_block,
            "target_x_axis": self._target_x_axis
        }

        return super().create_snapshot() | snapshot
    
    def restore_snapshot(self, snapshot: Any) -> PhysicalObservation:
        super().restore_snapshot(snapshot)

        self._subgoal_idx = snapshot["subgoal_idx"]
        self._subgoals = snapshot["subgoals"]
        self._last_dist_to_goal = snapshot["last_dist_to_goal"]
        self._last_collider_pos = snapshot["last_collider_pos"]
        self._last_dist_to_closest = snapshot["last_dist_to_closest"]
        self._last_eeff_x_axis = snapshot["last_eeff_x_axis"]
        self._steps_in_goal = snapshot["steps_in_goal"]
        self._target_x_axis = snapshot["target_x_axis"]

        if self._magnetized_block is None and snapshot["magnetized_block"] is not None:
            self._magnetize_block(snapshot["magnetized_block"])

        return self._get_observation()
    
    def step(self, action):
        self._perform_action(action)
        reward, terminated, truncated, info = self._evaluate_last_transition()
        observation = self._get_observation()

        self._last_dist_to_goal = self._get_distance_to_goal()
        self._last_collider_pos = self._get_collider_pose()[0]
        self._last_dist_to_closest = self._get_closest_block_info()[2]
        self._last_eeff_x_axis = self._get_eeff_x_axis()

        return observation, reward, terminated, truncated, info
    
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[PhysicalObservation, Dict[str, Any]]:
        if self._magnetized_block is not None:
            self._pbc.removeConstraint(self._magnetization_constraint)
        
        self._subgoal_idx = 0
        self._steps_in_goal = 0
        self._magnetized_block = None
        
        info = super().reset(seed=seed, options=options)[1]

        self._subgoals = self._generate_random_move()
        self._home_pos = self._pbc.getLinkState(self._robot, self._eeff)[0]
        self._home_x_axis = self._get_eeff_x_axis()
        self._activate_subgoal(self._subgoals[self._subgoal_idx])

        self._last_dist_to_goal = self._get_distance_to_goal()
        self._last_collider_pos = self._get_collider_pose()[0]
        self._last_dist_to_closest = self._get_closest_block_info()[2]
        self._last_eeff_x_axis = self._get_eeff_x_axis()

        obs = self._get_observation()
        
        return obs, info
    

    def _get_keyboard_mag(self) -> bool:
        keypresses = self._pbc.getKeyboardEvents()

        return keypresses.get(65297, pb.KEY_WAS_RELEASED) == pb.KEY_IS_DOWN # UP

    def _get_keyboard_eeff_delta(self) -> NDArray:
        keypresses = self._pbc.getKeyboardEvents()
        delta_x = delta_y = delta_z = 0
        step = 0.001
        
        if keypresses.get(105, pb.KEY_WAS_RELEASED) == pb.KEY_IS_DOWN: # i
            delta_z += step
        if keypresses.get(107, pb.KEY_WAS_RELEASED) == pb.KEY_IS_DOWN: # k
            delta_z -= step
        
        if keypresses.get(106, pb.KEY_WAS_RELEASED) == pb.KEY_IS_DOWN: # j
            delta_y -= step
        if keypresses.get(108, pb.KEY_WAS_RELEASED) == pb.KEY_IS_DOWN: # l
            delta_y += step
        
        if keypresses.get(117, pb.KEY_WAS_RELEASED) == pb.KEY_IS_DOWN: # u
            delta_x += step
        if keypresses.get(111, pb.KEY_WAS_RELEASED) == pb.KEY_IS_DOWN: # o
            delta_x -= step

        return np.array([delta_x, delta_y, delta_z])

    def step_human(self) -> Tuple[PhysicalObservation, float, bool, bool, Dict[str, Any]]:
        eeff_delta = np.array(self._get_keyboard_eeff_delta())

        if all(eeff_delta == np.array([0, 0, 0])):
            joints_action = [0] * len(self._active_joints) # [js[0] for js in self._pbc.getJointStates(self._robot, self._active_joints)]
        else:
            pos_eeff, orn_eeff = [np.array(v) for v in self._pbc.getLinkState(self._robot, self._eeff)[:2]]
            new_pos_eeff = pos_eeff + eeff_delta

            joints_action = self._pbc.calculateInverseKinematics(self._robot, self._eeff, new_pos_eeff, orn_eeff)

        # pos_goal, orn_goal = self._pbc.getBasePositionAndOrientation(self._goal)
        # joints_action = self._pbc.calculateInverseKinematics(self._robot, self._eeff, pos_goal)

        if self._get_keyboard_mag():
            self._magnetize_block("c")
        
        return self.step(joints_action)
