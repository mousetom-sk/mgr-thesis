from .base import Valuation, Proposition


class FuzzySemantics:

    true_treshold: float
    false_treshold: float
    ground_true: float
    ground_false: float

    def __init__(
        self, true_treshold: float = 0.7, false_treshold: float = 0.3,
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

    def is_true(self, proposition: Proposition, valuation: Valuation) -> bool:
        return valuation[proposition] >= self.true_treshold

    def is_false(self, proposition: Proposition, valuation: Valuation) -> bool:
        return valuation[proposition] <= self.false_treshold
    
    def set_true(self, proposition: Proposition, valuation: Valuation) -> bool:
        valuation[proposition] = self.ground_true
    
    def set_false(self, proposition: Proposition, valuation: Valuation) -> bool:
        valuation[proposition] = self.ground_false
