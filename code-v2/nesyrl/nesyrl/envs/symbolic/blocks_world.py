from typing import Tuple, List, Dict, Any

import copy
import itertools

from gymnasium import spaces

from nesyrl.envs.symbolic.base import SymbolicEnvironment, StateAtom, ActionAtom
from nesyrl.logic.propositional import Contradiction, Valuation, FuzzySemantics
from nesyrl.util import spaces as uspaces


class Block(StateAtom):

    def __init__(self, const: str) -> None:
        super().__init__(const)    


class On(StateAtom):

    def __init__(self, block1: str, block2: str) -> None:
        super().__init__(block1, block2)


class Top(StateAtom):

    def __init__(self, block: str) -> None:
        super().__init__(block)


class Move(ActionAtom):

    def __init__(self, block1: str, block2: str) -> None:
        super().__init__(block1, block2)


class BlocksWorld(SymbolicEnvironment):

    __constants__ = ["_table", "_horizon", "_use_top", "_goal_state", "_invalid_state"]

    _table = "table"

    _horizon: int
    _use_top: bool
    _blocks: List[str]
    _goal_state: Valuation
    _all_subgoals: List[Valuation]
    _initial_state: Valuation
    _invalid_state: Valuation
    _current_state: Valuation
    _current_step: int
    
    def __init__(
        self, horizon: int, blocks: List[str],
        goal_state: List[List[str]], initial_state: List[List[str]] | None = None,
        use_top: bool = True, semantics: FuzzySemantics = FuzzySemantics()
    ) -> None:
        super().__init__()

        if self._table in blocks:
            raise ValueError(f"No block can be named {self._table}")
        
        self.semantics = semantics
        
        self._horizon = horizon
        self._blocks = blocks[:]
        self._use_top = use_top
        self._init_state_atoms()
        self._init_action_atoms()

        self._goal_state = self._parse_raw_state(goal_state)
        self._initial_state = (None if initial_state is None
                               else self._parse_raw_state(initial_state))
        
        self._init_subgoals(goal_state)
        self._init_invalid_state()
        self._init_all_states()

        self.observation_space = spaces.Dict(
            {str(a): uspaces.Float(0, 1) for a in self.state_atoms}
        )
        self.action_space = spaces.Discrete(len(self.action_atoms))

    def _init_state_atoms(self) -> None:
        self.state_atoms = []
        self.domain_atoms = []

        for b1 in self._blocks + [self._table]:
            if self._use_top and b1 != self._table:
                self.domain_atoms.append(Block(b1))
                self.state_atoms.append(Block(b1))
                self.state_atoms.append(Top(b1))

            for b2 in self._blocks:
                if b1 != b2:
                    self.state_atoms.append(On(b2, b1))
        
        self.domain_atoms.append(Block(self._table))
        self.state_atoms.append(Block(self._table))
        self.state_atoms.append(Contradiction())
    
    def _init_action_atoms(self) -> None:
        self.action_atoms = []

        for b1 in self._blocks + [self._table]:
            for b2 in self._blocks:
                if b1 != b2:
                    self.action_atoms.append(Move(b2, b1))
    
    def _init_subgoals(self, raw_goal_state: List[List[str]]) -> None:
        blocks_backup = list(self._blocks)
        self._all_subgoals = []

        for stack in raw_goal_state:
            for i in range(1, len(stack)):
                substack = stack[:i]
                self._blocks = substack
                self._all_subgoals.append(self._parse_raw_state([substack]))

                if self._use_top:
                    self.semantics.set_false(Top(substack[-1]), self._all_subgoals[-1])

        self._blocks = blocks_backup 

    def _init_invalid_state(self) -> None:
        self._invalid_state = dict()

        for atom in self.state_atoms:
            self.semantics.set_false(atom, self._invalid_state)

        for atom in self.domain_atoms:
            self.semantics.set_true(atom, self._invalid_state)

        self.semantics.set_true(Contradiction(), self._invalid_state)
        self.semantics.set_false(Block(self._table), self._invalid_state)

    def _init_all_states(self) -> None:
        all_states = set()
        
        for permutation in itertools.permutations(self._blocks):
            for stacking in itertools.product(range(2), repeat=len(self._blocks) - 1):
                stacking = list(stacking) + [1]
                raw_state = []
                stack = []

                for block, end_stack in zip(permutation, stacking):
                    stack.append(block)

                    if end_stack:
                        raw_state.append(tuple(stack))
                        stack = []
                
                all_states.add(tuple(sorted(raw_state)))

        self.all_states = [self._parse_raw_state([list(stack) for stack in s])
                           for s in all_states]
        self.all_states.append(self._invalid_state)

    def _generate_random_state(self) -> Valuation:
        state_index = self.np_random.integers(len(self.all_states) - 1)
        state = self.all_states[state_index]

        return state

    def _generate_initial_state(self) -> Valuation:
        state = self._initial_state
        invalid = state is None

        while invalid:
            state = self._generate_random_state()
            invalid = self.semantics.is_subsumed(self._goal_state, state)
        
        return copy.deepcopy(state)

    def _is_top(self, block: str) -> bool:
        if self._use_top:
            return self.semantics.is_true(Top(block), self._current_state)
        
        return all(self.semantics.is_false(On(b, block), self._current_state)
                   for b in self._blocks if b != block)

    def _is_valid_action(self, action: ActionAtom) -> bool:
        if isinstance(action, Move):
            block1, block2 = action.args

            return (self.semantics.is_false(On(block1, block2) ,self._current_state)
                    and self._is_top(block1)
                    and (block2 == self._table or self._is_top(block2)))
        
        raise ValueError(f"Unknown action {action}")
    
    def _perform_action(self, action: ActionAtom) -> None:
        if not self._is_valid_action(action):
            self._current_state = self._invalid_state
            return
        
        self._current_step += 1
        block1, block2 = action.args

        for below in [self._table] + self._blocks:
            if below == block1:
                continue

            if self.semantics.is_true(On(block1, below), self._current_state):
                self.semantics.set_false(On(block1, below), self._current_state)

                if self._use_top and below != self._table:
                    self.semantics.set_true(Top(below), self._current_state)

                break

        self.semantics.set_true(On(block1, block2), self._current_state)

        if self._use_top and block2 != self._table:
            self.semantics.set_false(Top(block2), self._current_state)
    
    def _get_observation(self) -> Valuation:
        return copy.deepcopy(self._current_state)
    
    def _evaluate_last_transition(self) -> Tuple[float, bool, bool, Dict[str, Any]]:
        if self.semantics.is_true(Contradiction(), self._current_state):
            return -1.0, True, False, {"is_goal": False}
        
        is_goal = self.semantics.is_subsumed(self._goal_state, self._current_state)
        truncated = self._current_step >= self._horizon
        reward = 1.0 if is_goal else -0.1

        return reward, is_goal, truncated, {"is_goal": is_goal}

    def get_raw_state(self) -> List[List[str]]:
        remainig_blocks = set(self._blocks)
        raw_state = []

        for b in self._blocks:
            if self._is_top(b):
                raw_state.append([b])
                remainig_blocks.discard(b)

        while remainig_blocks:
            new_raw_state = []

            for stack in raw_state:
                for b in remainig_blocks:
                    if self.semantics.is_true(On(stack[0], b), self._current_state):
                        new_raw_state.append([b] + stack)
                        remainig_blocks.discard(b)
                        break
                else:
                    new_raw_state.append(stack)
            
            raw_state = new_raw_state

        return raw_state
        
    def _parse_raw_state(self, raw_state: List[List[str]]) -> Valuation:
        remaining_blocks = set(self._blocks)
        state = dict()

        for atom in self.state_atoms:
            self.semantics.set_false(atom, state)

        for atom in self.domain_atoms:
            self.semantics.set_true(atom, state)

        self.semantics.set_false(Block(self._table), state)

        for stack in raw_state:
            for b1, b2 in zip([self._table] + stack, stack + [None]):
                if b2 is None and b1 != self._table:
                    if self._use_top:
                        self.semantics.set_true(Top(b1), state)

                    continue

                if b2 not in remaining_blocks:
                    if b2 not in self._blocks:
                        raise ValueError(f"Unknown block {b2}")
                    
                    raise ValueError(f"Multiple occurences of the block {b2} in a single state.")
                
                self.semantics.set_true(On(b2, b1), state)
                remaining_blocks.discard(b2)

        if len(remaining_blocks) > 0:
            raise ValueError(f"These blocks are not positioned: {remaining_blocks}")
        
        return state
    
    def step(self, action: int) -> Tuple[Valuation, float, bool, Dict[str, Any]]:
        action = self.action_atoms[action]

        self._perform_action(action)
        observation = self._get_observation()
        reward, terminated, truncated, info = self._evaluate_last_transition()

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[Valuation, Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        if options is not None and "initial_state" in options:
            self._initial_state = self._parse_raw_state(options["initial_state"])

        self._current_step = 0
        self._current_state = self._generate_initial_state()
        observation = self._get_observation()

        return observation, {}
