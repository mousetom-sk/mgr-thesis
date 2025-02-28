from __future__ import annotations
from typing import Tuple, List, Dict


class PredicateAtom:
    
    __constants__ = ["args"]

    args: Tuple[str]

    def __init__(self, *args: str) -> None:
        self.args = args
    
    def __str__(self) -> str:
        args = (f"({', '.join(self.args)})" if len(self.args) > 0
                else "")

        return f"{self.__class__.__name__.lower()}{args}"
    
    def __hash__(self) -> int:
        return hash(str(self))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return str(self) == str(other)


class Contradiction(PredicateAtom):

    def __init__(self) -> None:
        super().__init__()


Valuation = Dict[str, float]


class Structure:
    
    __constants__ = ["domains", "valuation"]

    domains: Dict[str, List]
    valuation: Valuation

    def __init__(self, domains: Dict[str, List], valuation: Valuation) -> None:
        self.domains = domains
        self.valuation = valuation

    def __getitem__(self, atom: PredicateAtom | str) -> float:
        return self.valuation[str(atom)]
    
    def __setitem__(self, atom: PredicateAtom | str, value: float) -> float:
        self.valuation[str(atom)] = value

    @staticmethod
    def from_valuation(
        valuation: Valuation,
        domain_atoms: List[PredicateAtom | str],
        semantics: FuzzySemantics
    ) -> Structure:
        domain_atoms = [str(a) for a in domain_atoms]
        domains = dict()
        stripped = dict()

        for atom in valuation:
            if atom in domain_atoms:
                domain_name = atom.split("(")[0]

                if domain_name not in domains:
                    domains[domain_name] = []

                if semantics.is_true(atom, valuation):
                    domains[domain_name].append(atom.split("(")[1].split(")")[0])
            else:
                stripped[atom] = valuation[atom]
        
        return Structure(domains, stripped)


class FuzzySemantics:

    true_treshold: float
    false_treshold: float
    ground_true: float
    ground_false: float

    def __init__(
        self, true_treshold: float = 0.9, false_treshold: float = 0.1,
        ground_true: float = 1.0, ground_false: float = 0.0
    ) -> None:
        for val in [true_treshold, false_treshold, ground_true, ground_false]:
            if not (0.0 <= val <= 1.0):
                raise ValueError("All truth values must be within the range [0.0, 1.0]")
            
        if true_treshold < false_treshold:
            raise ValueError("true_treshold must be at least as high as false_treshold")
        
        if ground_true < true_treshold:
            raise ValueError("ground_true must be at least as high as true_treshold")
        
        if ground_false > false_treshold:
            raise ValueError("ground_false must be at most as high as false_treshold")
        
        self.true_treshold = true_treshold
        self.false_treshold = false_treshold
        self.ground_true = ground_true
        self.ground_false = ground_false

    def is_subsumed(self, child: Valuation, parent: Valuation) -> bool:
        return all(parent[atom] >= value for atom, value in child.items())

    def is_true(self, atom: PredicateAtom | str, valuation: Valuation) -> bool:
        return valuation[str(atom)] >= self.true_treshold

    def is_false(self, atom: PredicateAtom | str, valuation: Valuation) -> bool:
        return valuation[str(atom)] <= self.false_treshold
    
    def set_true(self, atom: PredicateAtom | str, valuation: Valuation) -> bool:
        valuation[str(atom)] = self.ground_true
    
    def set_false(self, atom: PredicateAtom | str, valuation: Valuation) -> bool:
        valuation[str(atom)] = self.ground_false
