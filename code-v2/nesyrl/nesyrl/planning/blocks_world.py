from typing import List, Dict, Any

import multiprocessing
import subprocess

import numpy as np

from tianshou.data import Batch
from tianshou.policy import BasePolicy

from nesyrl.envs.symbolic.blocks_world import BlocksWorld, ActionAtom, On, Move


class BlocksWorldASPPlanner(BasePolicy):

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

    _env: BlocksWorld
    _plan: List[ActionAtom]
    _plan_idx: int
    _log_dir: str

    def __init__(self, env: BlocksWorld, log_dir: str) -> None:
        super().__init__()

        self._env = env
        self._plan = []
        self._plan_idx = 0
        self._log_dir = log_dir

    def forward(self, *args, **kwargs) -> Batch:
        if self._plan_idx >= len(self._plan):
            self._horizon = self._env._horizon
            self._blocks = list(self._env._blocks)
            self._semantics = self._env.semantics

            self._goal_state = self._env._goal_state
            self._initial_state = self._env._current_state

            self._plan_idx = 0
            self._plan = self._solve()
        
        act = self._env.action_atoms.index(self._plan[self._plan_idx])
        self._plan_idx += 1

        return Batch(act=np.array([act]), state=None)

    def learn(self, batch: Batch, **kwargs) -> Dict[str, Any]:
        pass

    def _solve(self) -> List[ActionAtom]:
        program = self._generate_program()
        
        in_path = f"{self._log_dir}/blocks.lp"

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
                if b2 != b1 and self._semantics.is_true(On(b1, b2), self._initial_state):
                    asp.append(f"init(on({b1},{b2})).")
        
        asp.append("")

        for b1 in self._blocks:
            if self._semantics.is_true(On(b1, self._table), self._goal_state):
                asp.append(f"goal(on({b1},{self._table})).")

            for b2 in self._blocks:
                if b2 != b1 and self._semantics.is_true(On(b1, b2), self._goal_state):
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
