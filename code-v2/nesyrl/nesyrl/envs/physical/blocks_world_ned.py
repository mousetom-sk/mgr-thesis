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


class NedBlocksWorldBase(PhysicalEnvironment):

    metadata = {"render_modes": ["human"]}

    eeff_obs_key = "eeff"
    goal_obs_key = "goal"
    render_mode: str | None

    _table = "table"

    _horizon: int
    _current_step: int
    _current_blocks_state: List[List[str]]
    _all_blocks_states: List[List[List[str]]]
    _invalid_action = False

    _pbc: pbc.BulletClient
    _static_objects: Dict[str, int]
    _blocks: Dict[str, int]
    _goal: int
    _init_pbc_state: int
    
    _robot: int
    _eeff: int
    _eeff_name = "base_gripper"
    _active_joints: List[int]
    _active_joints_lb: NDArray
    _active_joints_ub: NDArray
    _active_joints_mf: NDArray
    _active_joints_mv: NDArray
    _max_force = 500
    _max_velocity = 0.5

    _velocity_epsilon = 1e-4
    _colision_epsilon = 1e-4

    _pb_data_dir = pbd.getDataPath()
    _models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")

    _static_objects_init = {
        "floor": {
            "fileName": os.path.join(_pb_data_dir, "plane.urdf"),
            "basePosition": [0, 0, 0],
            "baseOrientation": pb.getQuaternionFromEuler([0, 0, 0])
        },
        _table: { # width: 1.5, depth: 1; height: 0.645
            "fileName": os.path.join(_pb_data_dir, "table", "table.urdf"),
            "basePosition": [0.5, 0, 0],
            "baseOrientation": pb.getQuaternionFromEuler([0, 0, np.pi / 2])
        }
    }

    _block_init = {
        "fileName": os.path.join(_models_dir, "objects", "cube.urdf"),
        "globalScaling": 0.8
    }
    
    _block_colors = {
        "a": [1.0, 0.0, 0.0, 1.0],
        "b": [0.0, 1.0, 0.0, 1.0],
        "c": [0.0, 0.0, 1.0, 1.0],
        "d": [1.0, 0.8, 0.0, 1.0],
        "e": [0.5, 0.0, 1.0, 1.0],
        "f": [1.0, 0.0, 0.5, 1.0]
    }

    _robot_init = {
        "fileName": os.path.join(_models_dir, "robots", "ned", "niryo_ned.urdf"),
        "basePosition": [-0.1, 0, 0.65],
        "baseOrientation": pb.getQuaternionFromEuler([0, 0, 0])
    }

    _goal_init = {
        "fileName": os.path.join(_models_dir, "objects", "goal_area.urdf"),
        "globalScaling": 0.6
    }

    _goal_color = [0.0, 1.0, 0.0, 0.25]
    _transparent = [0.0, 1.0, 0.0, 0.0]

    _block_size = 0.05 * _block_init["globalScaling"]
    _blocks_x = 0.25
    _blocks_y_start = -0.14
    _blocks_y_delta = _block_size * 3 / 2
    _blocks_z_start = 0.645
    _blocks_z_delta = _block_size

    _goal_size = 0.05 * _goal_init["globalScaling"]
    _goal_shift = [- (_block_size + _goal_size) / 2,
                   0,
                   (_block_size - _goal_size) / 2]

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

        self._init_all_blocks_states()

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

    def _init_all_blocks_states(self) -> None:
        all_states = set()
        
        for permutation in itertools.permutations(self._blocks):
            for stacking in itertools.product(range(2), repeat=len(self._blocks) - 1):
                stacking = list(stacking) + [1]
                partial_state = []
                stack = []

                for block, end_stack in zip(permutation, stacking):
                    stack.append(block)

                    if end_stack:
                        partial_state.append(tuple(stack))
                        stack = []

                free_bases = len(self._block_colors) - len(partial_state)

                if free_bases == 0:
                    all_states.add(tuple(partial_state))
                    continue

                for ids in itertools.combinations(range(len(self._block_colors)), free_bases):
                    partial_copy = list(partial_state)
                    state = []

                    for i in range(len(self._block_colors)):
                        if i in ids:
                            state.append(tuple())
                        else:
                            state.append(partial_copy.pop())
                
                    all_states.add(tuple(state))

        self._all_blocks_states = [[list(stack) for stack in s]
                                   for s in all_states]
    
    def _load_active_joints_and_eeff(self, highlighting: bool) -> None:
        self._active_joints = []
        self._active_joints_lb = []
        self._active_joints_ub = []
        self._active_joints_mf = []
        self._active_joints_mv = []

        robot_joints = self._pbc.getNumJoints(self._robot)

        for id in range(robot_joints):
            joint_info = self._pbc.getJointInfo(self._robot, id)

            if self._eeff_name in str(joint_info[12]):
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
        # joints_state = np.array([self._pbc.getJointState(self._robot, id)[0]
        #                          for id in self._active_joints])
        # action *= np.pi
        # action /= 180
        
        # joints_range = self._active_joints_ub - self._active_joints_lb
        # joints_mid = self._active_joints_ub + self._active_joints_lb / 2
        # action *= joints_range / 2
        # action += joints_mid

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
                                    or self._is_moving_block()
                                    or self._is_robot_collision()
                                    # or self._is_behind_blocks()
                                    )
    
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

    def _is_moving_block(self) -> bool:
        velocities = list(self._pbc.getBaseVelocity(id)[0]
                          for id in self._blocks.values())
        
        return any(squared_length(v) >= (self._velocity_epsilon ** 2)
                   for v in velocities)

    def _is_robot_collision(self) -> bool:
        cps = self._pbc.getContactPoints(self._robot)
        
        return any(cp[8] < -self._colision_epsilon for cp in cps)
    
    def _is_behind_blocks(self) -> bool:
        pos_eeff = self._pbc.getLinkState(self._robot, self._eeff)[0]

        return pos_eeff[0] > self._blocks_x - self._block_size / 2

    def _generate_random_blocks_state(self) -> List[List[str]]:
        state_index = self.np_random.integers(len(self._all_blocks_states))
        state = self._all_blocks_states[state_index]

        return state
    
    def _set_blocks_state(self, state: List[List[str]]) -> None:
        orn = [0, 0, 0, 1]
        velocity = [0, 0, 0]
        bottom_x = self._blocks_x
        bottom_y = self._blocks_y_start
        bottom_z = self._blocks_z_start
        
        for stack in state:
            pos = np.array([bottom_x, bottom_y, bottom_z])

            for i, b in enumerate(stack):
                z_offset = i * np.array([0, 0, self._blocks_z_delta])

                self._pbc.resetBasePositionAndOrientation(self._blocks[b],
                                                          pos + z_offset,
                                                          orn)

                self._pbc.resetBaseVelocity(self._blocks[b], velocity, velocity)

            bottom_y += self._blocks_y_delta
    
    @abstractmethod
    def _evaluate_last_transition(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        pass

    def create_snapshot(self) -> Any:
        snapshot = {
            "current_step": self._current_step,
            "pbc_state": self._pbc.saveState()
        }

        return snapshot
    
    def restore_snapshot(self, snapshot: Any) -> PhysicalObservation:
        self._current_step = snapshot["current_step"]

        self._pbc.restoreState(snapshot["pbc_state"])

        return self._get_observation()
    
    def step(self, action: PhysicalAction) -> Tuple[PhysicalObservation, float, bool, bool, Dict[str, Any]]:
        self._perform_action(action)
        observation = self._get_observation()
        reward, terminated, truncated, info = self._evaluate_last_transition()
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[PhysicalObservation, Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self._pbc.restoreState(self._init_pbc_state)

        self._current_blocks_state = self._generate_random_blocks_state()
        self._set_blocks_state(self._current_blocks_state)

        self._current_step = 0
        observation = self._get_observation()
        self._invalid_action = False

        return observation, {}
        
    def close(self) -> None:
        self._pbc.disconnect()


class NedBlocksWorldMove(NedBlocksWorldBase):

    _home = "home"
    _subgoal_idx: int
    _subgoals: List[str]
    _last_dist_to_goal: float
    _last_eeff_pos: List[float]

    _steps_in_goal: int
    _achieved_treshold = 240

    _home_pos: List[float]

    def __init__(
        self, horizon: int, blocks: List[str], simulation_steps: int,
        highlighting: bool = True, render_mode: str | None = "human"
    ) -> None:
        super().__init__(horizon, blocks, simulation_steps, highlighting, render_mode)

        self._subgoal_idx = 0
        self._subgoals = []

    def _init_observation_space(self) -> None:
        size = 3 + 3 + 3 + len(self._active_joints)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(size,), dtype=float)

    def _get_observation(self) -> PhysicalObservation:
        pos_eeff, orn_eeff = self._pbc.getLinkState(self._robot, self._eeff)[:2]
        inv_pos_eeff, inv_orn_eeff = self._pbc.invertTransform(pos_eeff, orn_eeff)

        block_postions = []

        for b in self._blocks:
            # if self._subgoal_idx < len(self._subgoals) and b == self._subgoals[self._subgoal_idx]:
            #     continue

            pos, orn = self._pbc.getBasePositionAndOrientation(self._blocks[b])
            block_in_eeff = self._pbc.multiplyTransforms(inv_pos_eeff, inv_orn_eeff, pos, orn)
            block_postions.append((block_in_eeff[0], pos))

        pos_closest = sorted(block_postions, key=lambda x: squared_length(x[0]))[0][1]
        pos_goal = self._pbc.getBasePositionAndOrientation(self._goal)[0]

        joints_state = [self._pbc.getJointState(self._robot, id)[0]
                        for id in self._active_joints]

        observation = np.concatenate([np.array(pos_eeff),
                                      np.array(pos_goal),
                                      np.array(pos_closest),
                                      np.array(joints_state)])

        return observation

    def _generate_random_move(self) -> List[str]:
        subgoals = []
        valid_moves = []

        for i, stack1 in enumerate(self._current_blocks_state):
            if len(stack1) == 0:
                continue
            
            block = stack1[-1]
            on_table = len(stack1) == 1

            for j, stack2 in enumerate(self._current_blocks_state):
                if j == i or (on_table and len(stack2) == 0):
                    continue

                if len(stack2) == 0:
                    valid_moves.append((block, f"{self._table}{j}"))
                else:
                    valid_moves.append((block, stack2[-1]))

        subgoals.extend(self.np_random.choice(valid_moves))

        subgoals.append(self._home)

        return subgoals
    
    def _activate_subgoal(self, subgoal: str, extra_shift: bool = False) -> None:
        if subgoal in self._blocks:
            pos, orn = self._pbc.getBasePositionAndOrientation(self._blocks[subgoal])
            rel_pos, rel_orn = list(self._goal_shift), [0, 0, 0, 1]

            if extra_shift:
                rel_pos[2] += self._block_size

            goal_pos, goal_orn = self._pbc.multiplyTransforms(pos, orn, rel_pos, rel_orn)
        elif subgoal.startswith(self._table):
            i = int(subgoal[len(self._table):])

            pos = [self._blocks_x,
                   self._blocks_y_start + i * self._blocks_y_delta,
                   self._blocks_z_start]
            orn = [0, 0, 0, 1]
            rel_pos, rel_orn = list(self._goal_shift), [0, 0, 0, 1]

            goal_pos, goal_orn = self._pbc.multiplyTransforms(pos, orn, rel_pos, rel_orn)
        else:
            goal_pos, goal_orn = self._home_pos, [0, 0, 0, 1]

        self._pbc.resetBasePositionAndOrientation(self._goal, goal_pos, goal_orn)

    def _get_distance_to_goal(self) -> float:
        pos_eeff, orn_eeff = self._pbc.getLinkState(self._robot, self._eeff)[:2]
        inv_pos_eeff, inv_orn_eeff = self._pbc.invertTransform(pos_eeff, orn_eeff)

        pos_goal, orn_goal = self._pbc.getBasePositionAndOrientation(self._goal)
        goal_in_eeff = self._pbc.multiplyTransforms(inv_pos_eeff, inv_orn_eeff,
                                                    pos_goal, orn_goal)
        
        return length(goal_in_eeff[0])

    def _get_eeff_position(self) -> List[float]:
        return list(self._pbc.getLinkState(self._robot, self._eeff)[0])

    def _is_current_subgoal_achieved(self) -> bool:
        pos_goal = self._pbc.getBasePositionAndOrientation(self._goal)[0]
        pos_eeff = self._pbc.getLinkState(self._robot, self._eeff)[0]

        in_goal =  all(pos_goal[i] - self._goal_size / 2
                       < pos_eeff[i] <
                       pos_goal[i] + self._goal_size / 2
                       for i in range(3))
        
        if in_goal:
            self._steps_in_goal += self._simulation_steps
        
        return self._steps_in_goal > self._achieved_treshold

    def _move_block(self, block: str, to: str) -> None:
        orn = [0, 0, 0, 1]
        velocity = [0, 0, 0]
        
        if to in self._blocks:
            pos, orn = self._pbc.getBasePositionAndOrientation(self._blocks[to])
            rel_pos, rel_orn = [0, 0, self._blocks_z_delta], [0, 0, 0, 1]

            pos, orn = self._pbc.multiplyTransforms(pos, orn, rel_pos, rel_orn)
        else:
            i = int(to[len(self._table):])

            pos = [self._blocks_x,
                   self._blocks_y_start + i * self._blocks_y_delta,
                   self._blocks_z_start]

        self._pbc.resetBasePositionAndOrientation(self._blocks[block], pos, orn)
        self._pbc.resetBaseVelocity(self._blocks[block], velocity, velocity)

    def _evaluate_movement_to_goal(self) -> float:
        dist_to_goal = self._get_distance_to_goal()
        
        return self._last_dist_to_goal - dist_to_goal
    
    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def _evaluate_movement_to_extremes(self) -> float:
        eeff_pos = self._get_eeff_position()
        direction = np.array(eeff_pos) - np.array(self._last_eeff_pos)

        delta_x = -direction[0]
        factor_x = self._sigmoid(50 * (eeff_pos[0] - self._blocks_x - self._block_size / 2))

        delta_z = direction[2]
        factor_z = self._sigmoid(50 * (self._blocks_z_start - eeff_pos[2]))
        
        return min(factor_x * delta_x, factor_z * delta_z) / 10
    
    def _evaluate_last_transition(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        if self._invalid_action:
            return -0.1, True, False, {"is_goal": False}
        
        delta = self._evaluate_movement_to_goal()
        # delta += self._evaluate_movement_to_extremes()
        
        is_goal = self._subgoal_idx >= len(self._subgoals)
        truncated = self._current_step >= self._horizon
        reward = 1.0 if is_goal else delta
        
        return reward, is_goal, truncated, {"is_goal": is_goal}

    def create_snapshot(self):
        return super().create_snapshot() | {"subgoals": list(self._subgoals)}
    
    def restore_snapshot(self, snapshot):
        super().restore_snapshot(snapshot)

        self._subgoals = list(snapshot["subgoals"])

        return self._get_observation()
    
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        
        if self._is_current_subgoal_achieved():
            self._subgoal_idx += 1
            self._steps_in_goal = 0

            if self._subgoal_idx < len(self._subgoals):
                self._activate_subgoal(self._subgoals[self._subgoal_idx], True)
            
            if self._subgoal_idx == len(self._subgoals) - 1:
                self._move_block(self._subgoals[0], self._subgoals[1])

        self._last_dist_to_goal = self._get_distance_to_goal()
        self._last_eeff_pos = self._get_eeff_position()

        return observation, reward, terminated, truncated, info
    
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[PhysicalObservation, Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        self._subgoal_idx = 0
        self._subgoals = self._generate_random_move()
        self._steps_in_goal = 0
        self._home_pos = self._pbc.getLinkState(self._robot, self._eeff)[0]

        self._activate_subgoal(self._subgoals[self._subgoal_idx])
        self._last_dist_to_goal = self._get_distance_to_goal()
        self._last_eeff_pos = self._get_eeff_position()
        
        return obs, info
    

    
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
        
        return self.step(joints_action)
