from typing import Tuple, List, Dict, Any
from numpy.typing import NDArray

import os

import numpy as np
import pybullet as pb
import pybullet_utils.bullet_client as pbc
import pybullet_data as pbd
from gymnasium import spaces

from .base import PhysicalEnvironment, PhysicalAction, PhysicalObservation
from .util import *


class SimulatedNicoBlocksWorld(PhysicalEnvironment):

    metadata = {"render_modes": ["human"]}

    eeff_obs_key = "eeff"
    render_mode: str | None

    _table = "table"

    _horizon: int
    _goal_state: List[List[str]]
    _initial_state: List[List[str]]
    _current_step: int
    _was_action_valid: bool

    _pbc: pbc.BulletClient
    _static_objects: Dict[str, int]
    _blocks: Dict[str, int]
    _init_pbc_state: int
    
    _robot: int
    _eeff: int
    _eeff_name = "right_palm:11"
    _active_joints: List[int]
    _active_joints_lb: List[float]
    _active_joints_ub: List[float]
    _active_joints_mf: List[float]
    _active_joints_mv: List[float]

    _grasped_block: int | None = None
    _grasp_constraint: int

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
        "fileName": os.path.join(_models_dir, "robots", "nico", "upper_right_arm_only_fixed_hand.urdf"),
        "basePosition": [-0.1, 0, 0.65],
        "baseOrientation": pb.getQuaternionFromEuler([0, 0, 0])
    }

    def __init__(
        self, horizon: int, blocks: List[str],
        goal_state: List[List[str]], initial_state: List[List[str]] | None = None,
        render_mode: str | None = "human"
    ) -> None:
        super().__init__()

        if self.eeff_obs_key in blocks:
            raise ValueError(f"No block can be named {self.eeff_obs_key}")
        
        if self._table in blocks:
            raise ValueError(f"No block can be named {self._table}")
        
        self._horizon = horizon
        self._goal_state = goal_state
        self._initial_state = initial_state
        self.render_mode = render_mode

        self._init_pybullet()
        self._init_static_objects()
        self._init_blocks(blocks)
        self._init_robot()

        self._load_active_joints_and_eeff()
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
    
    def _init_robot(self) -> None:
        self._robot = self._pbc.loadURDF(useFixedBase=True,
                                         flags=pb.URDF_MERGE_FIXED_LINKS | pb.URDF_USE_SELF_COLLISION,
                                         **self._robot_init)

    def _load_active_joints_and_eeff(self) -> None:
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

            if joint_info[2] == pb.JOINT_FIXED:
                continue

            self._active_joints.append(id)
            self._active_joints_lb.append(joint_info[8])
            self._active_joints_ub.append(joint_info[9])
            self._active_joints_mf.append(joint_info[10])
            self._active_joints_mv.append(joint_info[11])
        
        self._active_joints_lb = np.array(self._active_joints_lb)
        self._active_joints_ub = np.array(self._active_joints_ub)
        self._active_joints_mf = np.array(self._active_joints_mf)
        self._active_joints_mv = np.array(self._active_joints_mv)

    def _init_action_space(self) -> None:
        self.action_space = spaces.OneOf([
            spaces.Discrete(2),
            spaces.Box(self._active_joints_lb, self._active_joints_ub, dtype=float)
        ])

    def _init_observation_space(self) -> None:
        # self.observation_space = spaces.Dict(
        #     {b: spaces.Box(-np.inf, np.inf, shape=(7,), dtype=float) for b in self._blocks}
        #     | {self.eeff_obs_key: spaces.Box(-np.inf, np.inf, shape=(7,), dtype=float)}
        # )

        self.observation_space = spaces.Dict(
            {b: spaces.Box(-np.inf, np.inf, shape=(7,), dtype=float) for b in self._blocks}
            | {f"joint_{id}": spaces.Box(-np.inf, np.inf, dtype=float) for id in self._active_joints}
            | {self.eeff_obs_key: spaces.Box(-np.inf, np.inf, shape=(7,), dtype=float)}
        )

    def _perform_action(self, action: PhysicalAction) -> None:
        if action[0] == 0:
            self._perform_gripper_action(action[1])
        else:
            self._perform_joints_action(action[1])

        self._current_step += 1
        self._pbc.stepSimulation()
    
    def _perform_joints_action(self, action: PhysicalAction) -> None:
        # joints_state = np.array([self._pbc.getJointState(self._robot, id)[0]
        #                          for id in self._active_joints])
        # action *= np.pi
        # action /= 180
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
        
        # self._pbc.setJointMotorControlArray(self._robot,
        #                                     self._active_joints,
        #                                     pb.POSITION_CONTROL,
        #                                     targetPositions=action,
        #                                     forces=self._active_joints_mf,
        #                                     positionGains=[0.7]*len(self._active_joints),
        #                                     velocityGains=[0.3]*len(self._active_joints))

    def _perform_gripper_action(self, action: int) -> None:
        if action == 0:
            self._perform_grasp()
        else:
            self._perform_release()

    def _perform_grasp(self) -> None:
        if self._grasped_block is not None:
            self._was_action_valid = False
            return
        
        block_id = self._find_closest_graspable_block()

        if block_id is None:
            self._was_action_valid = False
            return
        
        self._grasp_block(block_id)

    def _find_closest_graspable_block(self) -> int | None:
        pos_eeff, orn_eeff = self._pbc.getLinkState(self._robot, self._eeff)[:2]
        inv_pos_eeff, inv_orn_eeff = self._pbc.invertTransform(pos_eeff, orn_eeff)

        graspable = {}

        for block, id in self._blocks.items():
            pos, orn = self._pbc.getBasePositionAndOrientation(id)
            block_in_eeff = np.array(
                self._pbc.multiplyTransforms(inv_pos_eeff, inv_orn_eeff, pos, orn)[0]
            )

            if self._is_top(block) and self._can_grasp(block_in_eeff):
                graspable[id] = squared_length(block_in_eeff)

        if len(graspable) == 0:
            return None

        return sorted(graspable, key=lambda b: graspable[b])[0]
    
    def _grasp_block(self, block_id: int) -> None:
        self._grasped_block = block_id

        pos_eeff, orn_eeff = self._pbc.getLinkState(self._robot, self._eeff)[:2]
        rel_pos, rel_orn = [-0.025, 0.045, -0.035], [0, 0, 0, 1]
        pos, orn = self._pbc.multiplyTransforms(pos_eeff, orn_eeff, rel_pos, rel_orn)
        
        self._pbc.resetBasePositionAndOrientation(block_id, pos, orn)
        self._grasp_constraint = self._pbc.createConstraint(
            self._robot, self._eeff, block_id, -1, pb.JOINT_FIXED,
            [0, 0, 0], rel_pos, [0, 0, 0], rel_orn, [0, 0, 0, 1]
        )

    def _perform_release(self) -> None:
        if self._grasped_block is None:
            self._was_action_valid = False
            return
        
        table_position = self._find_closest_table_position()

        if table_position is not None:
            self._place_grasped_block_on_table(table_position)
            return

        block_id = self._find_closest_target_block()

        if block_id is None:
            self._was_action_valid = False
            return
        
        self._stack_grasped_block(block_id)

    def _find_closest_table_position(self) -> NDArray | None:
        ray_start, orn = self._pbc.getLinkState(self._robot, self._eeff)[:2]
        ray_end = self._pbc.multiplyTransforms(ray_start, orn, [0, 0.15, 0], [0, 0, 0, 1])[0]
        obj = self._pbc.rayTest(ray_start, ray_end)[0]

        if obj[0] == self._static_objects["table"]:
            return np.array(obj[3])
        
        return None

    def _place_grasped_block_on_table(self, position: NDArray) -> None:
        pos = position + np.array([0, 0, 0.02])
        orn = [0, 0, 0, 1]

        self._pbc.resetBasePositionAndOrientation(self._grasped_block, pos, orn)
        self._pbc.removeConstraint(self._grasp_constraint)
        self._grasped_block = None

    def _find_closest_target_block(self) -> int | None:
        pos_eeff, orn_eeff = self._pbc.getLinkState(self._robot, self._eeff)[:2]
        inv_pos_eeff, inv_orn_eeff = self._pbc.invertTransform(pos_eeff, orn_eeff)

        target = {}

        for block, id in self._blocks.items():
            if id == self._grasped_block:
                continue

            pos, orn = self._pbc.getBasePositionAndOrientation(id)
            block_in_eeff = np.array(
                self._pbc.multiplyTransforms(inv_pos_eeff, inv_orn_eeff, pos, orn)[0]
            )

            if self._is_top(block) and self._can_stack(block_in_eeff):
                target[id] = squared_length(block_in_eeff)

        if len(target) == 0:
            return None

        return sorted(target, key=lambda b: target[b])[0]
    
    def _stack_grasped_block(self, block_id: int) -> None:
        base_pos, base_orn = self._pbc.getBasePositionAndOrientation(block_id)
        rel_pos, rel_orn = [0, 0, 0.04], [0, 0, 0, 1]
        pos, orn = self._pbc.multiplyTransforms(base_pos, base_orn, rel_pos, rel_orn)

        self._pbc.resetBasePositionAndOrientation(block_id, pos, orn)
        self._pbc.removeConstraint(self._grasp_constraint)
        self._grasped_block = None

    def _dist_to_gripper(self, block: str) -> float:
        pos_eeff, orn_eeff = self._pbc.getLinkState(self._robot, self._eeff)[:2]
        rel_pos, rel_orn = [-0.025, 0.045, -0.035], [0, 0, 0, 1] # [-0.025, 0.08, -0.04], [0, 0, 0, 1]
        pos, orn = self._pbc.multiplyTransforms(pos_eeff, orn_eeff, rel_pos, rel_orn)
        inv_pos, inv_orn = self._pbc.invertTransform(pos, orn)

        id = self._blocks[block]
        pos, orn = self._pbc.getBasePositionAndOrientation(id)
        block_in_eeff = self._pbc.multiplyTransforms(inv_pos, inv_orn, pos, orn)

        dist = length(np.array(block_in_eeff[0]))
        
        # if not (block_in_eeff[2] < 0 and block_in_eeff[1] > 0):
        #     dist *= 10

        return dist

    def _can_grasp(self, block_in_eeff: NDArray) -> bool:
        return (squared_length(block_in_eeff) < (0.09 ** 2)
                and block_in_eeff[2] < 0 and block_in_eeff[1] > 0
                and -0.05 < block_in_eeff[0] < 0)
    
    def _can_stack(self, block_in_eeff: NDArray) -> bool:
        return (squared_length(block_in_eeff) < (0.15 ** 2)
                and block_in_eeff[2] < 0 and block_in_eeff[1] > 0)
    
    def _get_keyboard_gripper(self) -> int:
        keypresses = self._pbc.getKeyboardEvents()
        
        if keypresses.get(65297, pb.KEY_IS_DOWN) == pb.KEY_WAS_RELEASED: # UP
            return 0
        
        if keypresses.get(65298, pb.KEY_IS_DOWN) == pb.KEY_WAS_RELEASED: # DOWN
            return 1
        
        return -1
    
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
    
    def _get_observation(self) -> PhysicalObservation:
        observation = {}
        
        pos_eeff, orn_eeff = self._pbc.getLinkState(self._robot, self._eeff)[:2]
        rel_pos, rel_orn = [-0.025, 0.045, -0.035], [0, 0, 0, 1]
        pos_eeff, orn_eeff = self._pbc.multiplyTransforms(pos_eeff, orn_eeff, rel_pos, rel_orn)
        inv_pos_eeff, inv_orn_eeff = self._pbc.invertTransform(pos_eeff, orn_eeff)

        observation[self.eeff_obs_key] = np.concatenate([np.array(v) for v in (pos_eeff, orn_eeff)])

        for b in self._blocks:
            pos, orn = self._pbc.getBasePositionAndOrientation(self._blocks[b])
            block_in_eeff = self._pbc.multiplyTransforms(inv_pos_eeff, inv_orn_eeff, pos, orn)
            observation[b] = np.concatenate([np.array(v) for v in block_in_eeff])
        
        for id in self._active_joints:
            joint_state = self._pbc.getJointState(self._robot, id)
            observation[f"joint_{id}_pos"] = joint_state[0]
            # observation[f"joint_{id}_vel"] = joint_state[1]

        # pos_orn = self._pbc.getLinkState(self._robot, self._eeff)[:2]
        # observation[self.eeff_obs_key] = np.concatenate([np.array(v) for v in pos_orn])

        return observation

    def _is_moving_block(self) -> bool:
        velocities = list(self._pbc.getBaseVelocity(id)[0]
                          for id in self._blocks.values()
                          if (self._grasped_block is None or id != self._grasped_block))
        
        return any(squared_length(v) >= (0.0001 ** 2) for v in velocities)

    def _is_robot_collision(self) -> bool:
        cps = self._pbc.getContactPoints(self._robot)
        cps = list(filter(lambda cp: cp[8] < 0, cps))
    
        if self._grasped_block is None:
            return len(cps) > 0
        
        cps = list(filter(lambda cp: cp[2] != self._grasped_block, cps))
        
        if len(cps) > 0:
            return True
        
        cps = self._pbc.getContactPoints(self._grasped_block)
        cps = list(filter(lambda cp: cp[8] < 0 and cp[2] != self._robot, cps))
        
        return len(cps) > 0
    
    def _is_goal(self) -> bool:
        for stack in self._goal_state:
            if not self._is_on(stack[0], self._table):
                return False
            
            for above, below in zip(stack[1:], stack):
                if not self._is_on(above, below):
                    return False

        return True
    
    def _is_top(self, block: str) -> bool:
        return all(not self._is_on(other, block)
                   for other in self._blocks if other != block)

    def _is_on(self, above: str, below: str) -> bool:
        if below == self._table:
            cps = self._pbc.getContactPoints(self._blocks[above], self._static_objects[self._table])
        else:
            cps = self._pbc.getContactPoints(self._blocks[above], self._blocks[below])
        
        cps = list(filter(lambda cp: cp[8] < 0, cps))
        
        if len(cps) == 0:
            return False
        
        return all(is_main_component(np.array(cp[7]), SignedComponent.Z_POS)
                   for cp in cps)
    
    def _generate_initial_blocks_state(self) -> List[List[str]]:
        state = self._initial_state
        invalid = state is None

        while invalid:
            state = self._generate_random_blocks_state()
            invalid = self._goal_state == state
        
        return list(state)

    def _generate_random_blocks_state(self) -> List[List[str]]:
        shuffled_blocks = self.np_random.permutation(list(self._blocks))
        stack_ends = np.concatenate((self.np_random.integers(2, size=len(self._blocks) - 1),
                                     np.array([1])))

        blocks_state = []
        stack = []

        for (block, end_stack) in zip(shuffled_blocks, stack_ends):
            stack.append(block)

            if end_stack:
                blocks_state.append(stack)
                stack = []

        return blocks_state
    
    def _set_blocks_state(self, state: List[List[str]]) -> None:
        bottom_x_space = spaces.Box(0.20, 0.23, dtype=float)
        bottom_y_start_space = spaces.Box(-0.1, -0.08, dtype=float)
        bottom_y_delta_space = spaces.Box(0.06, 0.08, dtype=float)

        orn = [0, 0, 0, 1]
        velocity = [0, 0, 0]
        bottom_x = bottom_x_space.sample()[0]
        bottom_y = bottom_y_start_space.sample()[0]
        bottom_z = 0.645
        
        for stack in state:
            pos = np.array([bottom_x, bottom_y, bottom_z])

            for i, b in enumerate(stack):
                z_offset = i * np.array([0, 0, 0.04])

                self._pbc.resetBasePositionAndOrientation(self._blocks[b],
                                                          pos + z_offset,
                                                          orn)

                self._pbc.resetBaseVelocity(self._blocks[b], velocity, velocity)
            
            bottom_x = bottom_x_space.sample()[0]
            bottom_y += bottom_y_delta_space.sample()[0]
    
    def _evaluate_last_transition(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        invalid = (not self._was_action_valid or self._is_moving_block()
                   or self._is_robot_collision())

        if invalid:
            return -1.0, True, False, {"is_goal": False, "is_valid": False}
    
        is_goal = self._is_goal()
        truncated = self._current_step >= self._horizon
        reward = 1.0 if is_goal else -0.01
        
        return reward, is_goal, truncated, {"is_goal": is_goal, "is_valid": True}

    def create_snapshot(self) -> Any:
        snapshot = {
            "grasped_block": self._grasped_block,
            "current_step": self._current_step,
            "was_action_valid": self._was_action_valid,
            "pbc_state": self._pbc.saveState()
        }

        return snapshot
    
    def restore_snapshot(self, snapshot: Any) -> PhysicalObservation:
        if self._grasped_block is not None:
            self._pbc.removeConstraint(self._grasp_constraint)
        
        if snapshot["grasped_block"] is not None:
            self._grasp_block(snapshot["grasped_block"])

        self._current_step = snapshot["current_step"]
        self._was_action_valid = snapshot["was_action_valid"]

        self._pbc.restoreState(snapshot["pbc_state"])

        return self._get_observation()
    
    def step(self, action: PhysicalAction) -> Tuple[PhysicalObservation, float, bool, bool, Dict[str, Any]]:
        self._perform_action(action)
        observation = self._get_observation()
        reward, terminated, truncated, info = self._evaluate_last_transition()
        
        return observation, reward, terminated, truncated, info
    
    def step_human(self) -> Tuple[PhysicalObservation, float, bool, bool, Dict[str, Any]]:
        gripper_action = self._get_keyboard_gripper()

        if gripper_action >= 0:
            return self.step([0, gripper_action])

        eeff_delta = np.array(self._get_keyboard_eeff_delta())

        if all(eeff_delta == np.array([0, 0, 0])):
            joints_action = [js[0] for js in self._pbc.getJointStates(self._robot, self._active_joints)]
        else:
            pos_eeff, orn_eeff = [np.array(v) for v in self._pbc.getLinkState(self._robot, self._eeff)[:2]]
            new_pos_eeff = pos_eeff + eeff_delta

            joints_action = self._pbc.calculateInverseKinematics(self._robot, self._eeff, new_pos_eeff, orn_eeff)
        
        return self.step([1, joints_action])
    
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[PhysicalObservation, Dict[str, Any]]:
        super().reset(seed=seed)

        if self._grasped_block is not None:
            self._pbc.removeConstraint(self._grasp_constraint)

        self._pbc.restoreState(self._init_pbc_state)

        state = self._generate_initial_blocks_state()
        self._set_blocks_state(state)

        self._current_step = 0
        self._was_action_valid = True
        self._grasped_block = None
        observation = self._get_observation()

        return observation, {}
        
    def close(self) -> None:
        self._pbc.disconnect()
