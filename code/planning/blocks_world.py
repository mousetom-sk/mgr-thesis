from typing import List

import multiprocessing
import subprocess

from envs.symbolic import Valuation, ActionAtom, On, Move
from envs.symbolic.util import FuzzySemantics


class BlocksWorldASPPlanner:

    _table = "table"

    _asp_step = f"""
#program step(t).

{{move(X,Y,t) : block(X), block(Y), X != Y;
 move(X,{_table},t) : block(X)}} = 1.

:- move(X,Y,t), holds(on(X,Y),t-1).
:- move(X,Y,t), holds(on(Z,X),t-1).
:- move(X,Y,t), holds(on(Z,Y),t-1), block(Y).

holds(on(X,Y),t) :- move(X,Y,t), block(X).

moved(X,t) :- move(X,Y,t), block(X).
holds(on(X,Y),t) :- holds(on(X,Y),t-1), not moved(X,t), block(X)."""
    
    _asp_check = """
#program check(t).

:- query(t), goal(F), not holds(F,t).

#show move/3."""

    _horizon: int
    _semantics: FuzzySemantics
    _blocks: List[str]
    _goal_state: Valuation
    _initial_state: Valuation

    def __init__(
        self, horizon: int, blocks: List[str],
        goal_state: List[List[str]], initial_state: List[List[str]],
        semantics: FuzzySemantics = FuzzySemantics()
    ) -> None:
        super().__init__()

        if self._table in blocks:
            raise ValueError(f"No block can be named {self._table}")
        
        self._horizon = horizon
        self._blocks = list(blocks)
        self._semantics = semantics

        self._goal_state = self._parse_raw_state(goal_state)
        self._initial_state = self._parse_raw_state(initial_state)
    
    def _parse_raw_state(self, raw_state: List[List[str]]) -> Valuation:
        remaining_blocks = set(self._blocks)
        state = {}

        for b1 in self._blocks:
            self._semantics.set_false(On(b1, self._table), state)

            for b2 in self._blocks:
                self._semantics.set_false(On(b1, b2), state)

        for stack in raw_state:
            for b1, b2 in zip([self._table] + stack, stack + [None]):
                if b2 is None and b1 != self._table:
                    continue

                if b2 not in remaining_blocks:
                    if b2 not in self._blocks:
                        raise ValueError(f"Unknown block {b2}")
                    
                    raise ValueError(f"Multiple occurences of the block {b2} in a single state.")
                
                self._semantics.set_true(On(b2, b1), state)
                remaining_blocks.discard(b2)

        if len(remaining_blocks) > 0:
            raise ValueError(f"These blocks are not positioned: {remaining_blocks}")
        
        return state

    def solve(self) -> List[ActionAtom]:
        program = self._generate_program()
        
        in_path = "planning/blocks.lp"

        with open(in_path, "w") as lp:
            print("\n".join(program), file=lp)

        workers_count = max(multiprocessing.cpu_count() // 2, 1)
        clingo_output = []

        with subprocess.Popen(
            ["clingo", "--parallel-mode", str(workers_count), in_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True
        ) as clingo:
            for line in clingo.stdout:
                line = line.rstrip()
                clingo_output.append(line)
        
        return self._read_plan(clingo_output)
    
    def _generate_program(self) -> List[str]:
        asp = [
            "#include <incmode>.",
            f"#const imax = {self._horizon + 1}.\n",
            f"block({';'.join(self._blocks)}).\n",
        ]

        for b1 in self._blocks:
            if self._semantics.is_true(On(b1, self._table), self._initial_state):
                asp.append(f"init(on({b1},{self._table})).")

            for b2 in self._blocks:
                if self._semantics.is_true(On(b1, b2), self._initial_state):
                    asp.append(f"init(on({b1},{b2})).")
        
        asp.append("")

        for b1 in self._blocks:
            if self._semantics.is_true(On(b1, self._table), self._goal_state):
                asp.append(f"goal(on({b1},{self._table})).")

            for b2 in self._blocks:
                if self._semantics.is_true(On(b1, b2), self._goal_state):
                    asp.append(f"goal(on({b1},{b2})).")

        asp.append("")
        asp.append("holds(F,0) :- init(F).")
        asp.append(self._asp_step)
        asp.append(self._asp_check)

        return asp
    
    def _read_plan(self, clingo_output: List[str]) -> List[ActionAtom]:
        plan = None
        was_ans = False

        for line in clingo_output:
            line = line.strip()

            if was_ans:
                plan = []

                for op in line.split():
                    x, y, _ = op.split(",")
                    x = x.split("(")[1]

                    plan.append(Move(x, y))

                break

            if line.startswith('UNSAT'):
                break
            
            if line.startswith('Ans'):
                was_ans = True

        return plan
