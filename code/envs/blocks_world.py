from __future__ import annotations
from typing import List, Dict, Tuple, Optional

import numpy as np

from lib import Environment


class BlocksWorld(Environment):

    class Proposition:

        def __init__(self, name: str, *args: str) -> None:
            self._name = name
            self._args = args
        
        def __str__(self) -> str:
            args = (f"({', '.join(self._args)})" if len(self._args) > 0
                    else "")

            return f"{self._name}{args}"

    class StateAtom(Environment.Feature):
        
        def __init__(self, name: str, *args: str) -> None:
            super().__init__()

            self._proposition = BlocksWorld.Proposition(name, *args)

        def __str__(self) -> str:
            return str(self._proposition)
        
        def __hash__(self) -> int:
            return hash(str(self))
        
        def __eq__(self, other: object) -> bool:
            if not isinstance(other, self.__class__):
                return False

            return str(self) == str(other)
        
    class On(StateAtom):

        def __init__(self, block1: str, block2: str) -> None:
            super().__init__("on", block1, block2)
    
    class Top(StateAtom):

        def __init__(self, block: str) -> None:
            super().__init__("top", block)
    
    class State(Environment.State):
        
        def __init__(self, valuation: Dict[BlocksWorld.StateAtom, float]):
            super().__init__()

            self._valuation = valuation

        @property
        def features(self) -> Dict[BlocksWorld.StateAtom, float]:
            return self._valuation
        
        def is_subset(self, other: BlocksWorld.State) -> bool:
            return all(other._valuation[atom] >= value
                       for (atom, value) in self._valuation.items())
        
        def __hash__(self) -> int:
            return hash(tuple(self._valuation.items()))
        
        def __eq__(self, other: object) -> bool:
            if not isinstance(other, self.__class__):
                return False

            return self._valuation == other._valuation
        
    class ActionAtom(Environment.Action):

        def __init__(self, name: str, *args: str) -> None:
            super().__init__()

            self._proposition = BlocksWorld.Proposition(name, *args)

        def __str__(self) -> str:
            return str(self._proposition)
        
        def __hash__(self) -> int:
            return hash(str(self))
        
        def __eq__(self, other: object) -> bool:
            if not isinstance(other, self.__class__):
                return False

            return str(self) == str(other)
        
    class Move(ActionAtom):

        def __init__(self, block1: str, block2: str) -> None:
            super().__init__("move", block1, block2)

            self.block1 = block1
            self.block2 = block2

    
    TABLE: str = "table"

    def __init__(self, blocks: List[str], goal_state: List[List[str]], initial_state: Optional[List[List[str]]] = None) -> None:
        super().__init__()

        if self.TABLE in blocks:
            raise ValueError(f"No block can be named {self.TABLE}")

        self._blocks = blocks[:]
        self._state_atoms = self._generate_state_atoms()
        self._action_atoms = self._generate_action_atoms()

        self._goal_state = self._parse_raw_state(goal_state)
        self._initial_state = (self._parse_raw_state(initial_state) if initial_state is not None
                               else None)
        
        if self._initial_state:
            self._subgoals = self._generate_subgoals(goal_state)
        
        self.reset()

    def _generate_state_atoms(self) -> List[StateAtom]:
        atoms = []

        for b1 in self._blocks + [self.TABLE]:
            if b1 != self.TABLE:
                atoms.append(self.Top(b1))

            for b2 in self._blocks:
                if b1 != b2:
                    atoms.append(self.On(b2, b1))

        atoms.append(self.On(self._blocks[0], self._blocks[0]))
        
        return atoms
    
    def _generate_action_atoms(self) -> List[ActionAtom]:
        atoms = []

        for b1 in self._blocks + [self.TABLE]:
            for b2 in self._blocks:
                if b1 != b2:
                    atoms.append(self.Move(b2, b1))
        
        return atoms
    
    def _generate_subgoals(self, raw_goal_state: List[List[str]]) -> List[State]:
        blocks_backup = list(self._blocks)
        subgoals = []

        for stack in raw_goal_state:
            for i in range(1, len(stack)):
                substack = stack[:i]
                self._blocks = substack
                subgoals.append(self._parse_raw_state([substack]))

                subgoals[-1]._valuation[self.Top(substack[-1])] = 0

        self._blocks = blocks_backup
        subgoals = list(filter(lambda sg: not sg.is_subset(self._initial_state), subgoals))

        return subgoals    

    def _parse_raw_state(self, raw_state: List[List[str]]) -> State:
        remaining_blocks = set(self._blocks)
        true_state_atoms = set()

        for stack in raw_state:
            for b1, b2 in zip([self.TABLE] + stack, stack + [None]):
                if b2 is None and b1 != self.TABLE:
                    true_state_atoms.add(self.Top(b1))
                    continue

                if b2 not in remaining_blocks:
                    if b2 not in self._blocks:
                        raise ValueError(f"Unknown block {b2}")
                    
                    raise ValueError(f"Multiple occurences of the block {b2} in a single state.")
                
                true_state_atoms.add(self.On(b2, b1))
                remaining_blocks.discard(b2)

        if len(remaining_blocks) > 0:
            raise ValueError(f"These blocks are not positioned: {b2}")
        
        valuation = dict((atom, int(atom in true_state_atoms)) for atom in self._state_atoms)
        return self.State(valuation)

    @property
    def feature_space(self) -> List[StateAtom]:
        return self._state_atoms

    @property
    def action_space(self) -> List[ActionAtom]:
        return self._action_atoms

    def is_final(self) -> bool:
        return self._bad_action or self.is_goal()
    
    def is_goal(self) -> bool:
        return self._goal_state.is_subset(self._current_state) # validity: False

    def step(self, action: ActionAtom) -> Tuple[State, int]:
        print(action, end=" ")
        
        if not self.is_valid_action(action):
            self._bad_action = True
            return self._current_state, -0.1 # 2
        
        if isinstance(action, self.Move):
            new_valuation = dict(self._current_state.features)

            for below in [self.TABLE] + self._blocks:
                if below == action.block1:
                    continue

                if new_valuation[self.On(action.block1, below)]:
                    new_valuation[self.On(action.block1, below)] = 0

                    if below != self.TABLE:
                        new_valuation[self.Top(below)] = 1

                    break

            new_valuation[self.On(action.block1, action.block2)] = 1

            if action.block2 != self.TABLE:
                new_valuation[self.Top(action.block2)] = 0

            self._current_state = self.State(new_valuation)

        if self._goal_state.is_subset(self._current_state):
            reward = 1
        else:
            reached = set(sg for sg in self._subgoals if sg.is_subset(self._current_state))
            subgoal_diff = len(reached - self._subgoals_reached) - 1.1 * len(self._subgoals_reached - reached)

            if subgoal_diff != 0:
                reward = subgoal_diff
            else:
                reward = -0.1

            self._subgoals_reached = reached

            # generic (option): reward = -0.1

        # validity: reward = 0.1

        return self._current_state, reward
    
    def is_valid_action(self, action: ActionAtom) -> bool:
        if isinstance(action, self.Move):
            valuation = self._current_state.features

            if valuation[self.On(action.block1, action.block2)]:
                return False
            
            if not valuation[self.Top(action.block1)]:
                return False
            
            if action.block2 != self.TABLE and not valuation[self.Top(action.block2)]:
                return False

            return True
        
        raise ValueError(f"Unknown action {action}")

    def reset(self) -> State:
        self._bad_action = False
        self._subgoals_reached = set()

        if self._initial_state is None:
            self._current_state = self._generate_random_state()
        else:
            self._current_state = self._initial_state

        return self._current_state

    def _generate_random_state(self) -> State:
        shuffled_blocks = np.random.permutation(self._blocks)
        stack_ends = np.concatenate((np.random.randint(2, size=len(self._blocks) - 1), np.array([1])))

        raw_state = []
        stack = []

        for (block, end_stack) in zip(shuffled_blocks, stack_ends):
            stack.append(block)

            if end_stack:
                raw_state.append(stack)
                stack = []

        return self._parse_raw_state(raw_state)
