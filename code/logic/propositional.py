from typing import Tuple, List, Dict


class PredicateAtom:
    
    __constants__ = ["args"]

    args: Tuple[str]

    def __init__(self, *args: str) -> None:
        self.args = args
    
    def __str__(self) -> str:
        args = (f"({', '.join(self.args)})" if len(self.args) > 0
                else "")

        return f"{self.__class__.__name__}{args}"
    
    def __hash__(self) -> int:
        return hash(str(self))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return str(self) == str(other)


class Contradiction(PredicateAtom):

    def __init__(self) -> None:
        super().__init__("FALSE")


class Structure:
    
    __constants__ = ["domains", "valuation"]

    domains: Dict[str, List]
    valuation: Dict[PredicateAtom, float]

    def __init__(self, domains: Dict[str, List], valuation: Dict[PredicateAtom, float]) -> None:
        self.domains = domains
        self.valuation = valuation

    def __getitem__(self, atom: str) -> float:
        return self.valuation[atom]
    
    def __setitem__(self, atom: str, value: float) -> float:
        self.valuation[atom] = value


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

    def is_subsumed(self, child: Structure, parent: Structure) -> bool:
        return all(parent[atom] >= value for atom, value in child.items())

    def is_true(self, atom: PredicateAtom, structure: Structure) -> bool:
        return structure[atom] >= self.true_treshold

    def is_false(self, atom: PredicateAtom, structure: Structure) -> bool:
        return structure[atom] <= self.false_treshold
    
    def set_true(self, atom: PredicateAtom, structure: Structure) -> bool:
        structure[atom] = self.ground_true
    
    def set_false(self, atom: PredicateAtom, structure: Structure) -> bool:
        structure[atom] = self.ground_false
