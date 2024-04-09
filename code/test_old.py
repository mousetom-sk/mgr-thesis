from lnn import Proposition, And, Fact, Or
from lnn import Model, Loss, Direction, Predicate


# Constants
A, B, C, D = list("abcd")
TABLE = "table"

# Predicates
# on = Predicate("on", 2)
# top = Predicate("top", 1)
# move = Predicate("move", 2)

# Rules
# moveBC = Proposition(f"move({B}, {C})")

onAB = Proposition(f"on({A}, {B})")
onBTABLE = Proposition(f"on({B}, {TABLE})")
onCTABLE = Proposition(f"on({C}, {TABLE})")
onDTABLE = Proposition(f"on({D}, {TABLE})")
topB = Proposition(f"top({B})")
topC = Proposition(f"top({C})")
topD = Proposition(f"top({D})")

# onAB = Proposition(f"on({A}, {B})")
# onBTABLE = Proposition(f"on({B}, {TABLE})")
onCD = Proposition(f"on({C}, {D})")
# onDTABLE = Proposition(f"on({D}, {TABLE})")
# topB = Proposition(f"top({B})")
# topC = Proposition(f"top({C})")

case1 = And(onAB, onBTABLE, onCTABLE, onDTABLE, topB, topC, topD)
case2 = And(onAB, onBTABLE, onCD, onDTABLE, topB, topC)
moveBC = Or(case1, case2)

# Data
onAB.add_data(Fact.TRUE)
onBTABLE.add_data(Fact.TRUE)
onCTABLE.add_data(Fact.TRUE)
onDTABLE.add_data(Fact.TRUE)
topB.add_data(Fact.TRUE)
topC.add_data(Fact.TRUE)
topD.add_data(Fact.TRUE)

# Knowledge
model = Model()
model.add_knowledge(moveBC)

# Reasoning
model.infer()
moveBC.print()

model.flush()
onAB = Proposition(f"on({A}, {B})")
onBTABLE = Proposition(f"on({B}, {TABLE})")
onCTABLE = Proposition(f"on({C}, {TABLE})")
onDTABLE = Proposition(f"on({D}, {TABLE})")
topB = Proposition(f"top({B})")
topC = Proposition(f"top({C})")
topD = Proposition(f"top({D})")
moveBC.print()

# model.add_labels({
#     AND: (0.8, 0.8)
# })

# model.train(direction=Direction.UPWARD, losses=[Loss.SUPERVISED])
# AND.print(params=True)
