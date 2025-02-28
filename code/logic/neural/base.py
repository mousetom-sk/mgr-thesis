from __future__ import annotations
from typing import Dict, Set, List, Tuple, Iterable, Callable

import itertools
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter

from .weights import (WeightInitializer,
                      WeightRegularizer, NoRegularizer,
                      WeightPostprocessor, NoPostprocessor)
from ..propositional import PredicateAtom, Structure


class NLVariable:
    
    name: str
    domain: str

    def __init__(self, name: str, domain: str) -> None:
        self.name = name
        self.domain = domain

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(str(self))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return str(self) == str(other)
    
    def __lt__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return str(self) < str(other)


class NLModule(ABC, Module):

    nl_modules: ModuleList
    last_output: Tensor

    _transforms: List[Callable[[Tensor, Structure], Tensor]]

    def get_dependencies(self) -> Dict[NLModule, Set[NLModule]]:
        dependencies = dict()
        
        for module in self.nl_modules:
            dependencies |= module.get_dependencies()

        return dependencies
    
    @abstractmethod
    def prepare_transforms(self) -> List[NLVariable]:
        pass

    def __hash__(self) -> int:
        return hash(str(self))


class NLPredicateAtom(NLModule):

    predicate: type[PredicateAtom]
    args: Tuple[NLVariable | str]

    _output_vars: List[NLVariable]
    _var_indices: Dict[NLVariable, int]
    
    def __init__(self, predicate: type[PredicateAtom], *args: NLVariable | str):
        super().__init__()

        self.predicate = predicate
        self.args = args

    def prepare_transforms(self) -> List[NLVariable]:
        var_args = set(arg for arg in self.args if isinstance(arg, NLVariable))
        self._output_vars = sorted(var_args)
        self._var_indices = dict((v, self._output_vars.index(v)) for v in var_args)

        return self._output_vars

    @abstractmethod
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        output = []
        
        for struct in structures:
            if len(self._output_vars) == 0:
                output.append([struct.valuation[self.predicate(*self.args)]])
                continue
        
            domains = [struct.domains[v.domain] for v in self._output_vars]
            outs = []

            for sub in itertools.product(*domains):
                apply_sub = lambda arg: (sub[self._var_indices[arg]]
                                         if isinstance(arg, NLVariable) else arg)
                args = map(apply_sub, self.args)

                outs.append(struct.valuation[self.predicate(*args)])

            output.append(torch.tensor(outs).reshape([len(d) for d in domains]))

        return torch.tensor(output)

    def __str__(self) -> str:
        if len(self.args) > 0:
            args = f"({', '.join(str(arg) for arg in self.args)})"
        else:
            args = ""

        return f"{self.predicate.symbol}{args}"


class NLModuleParametrized(NLModule):

    __constants__ = ["weight_initializer", "weight_regularizer",
                     "weight_postprocessor"]
    
    weight_initializer: WeightInitializer
    weight_regularizer: WeightRegularizer
    weight_postprocessor: WeightPostprocessor
    weight: Tensor

    def __init__(
        self, operands: List[NLModule],
        weight_initializer: WeightInitializer,
        weight_regularizer: WeightRegularizer = NoRegularizer(),
        weight_postprocessor: WeightPostprocessor = NoPostprocessor(),
        trainable: bool | Tensor = True
    ):
        super().__init__()

        self.nl_modules = ModuleList(operands)

        self.weight_initializer = weight_initializer
        self.weight_regularizer = weight_regularizer
        self.weight_postprocessor = weight_postprocessor

        if isinstance(trainable, bool):
            self.weight = Parameter(torch.empty((len(operands),)), requires_grad=trainable)
        else:
            self.weight = Parameter(torch.empty((len(operands),)))
            self.weight.register_hook(lambda grad: trainable * grad)

    def prepare_transforms(self) -> List[NLVariable]:

        def extend(x: Tensor, s: Structure, mask: Tensor) -> Tensor:
            full_shape = torch.tensor(
                [x.shape[0]] + [len(s.domains[v.domain]) for v in output_vars]
            )
        
            shape = torch.clone(full_shape)
            shape[~mask] = 1
            x.reshape(shape)

            full_shape[0] = 1
            full_shape[mask] = 1

            return torch.tile(x, full_shape)

        input_vars = [module.prepare_transforms()
                      for module in self.nl_modules]
        
        union = set()
        for vars in input_vars:
            union.update(vars)

        output_vars = sorted(union)

        self._transforms = []
        for vars in input_vars:
            mask = torch.isin(output_vars, torch.tensor(vars))
            self._transforms.append(lambda x, s: extend(x, s, mask))

        return output_vars
    
    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight_initializer.initialize(self.weight)
    
    def compute_regularization_loss(self) -> Tensor:
        return self.weight_regularizer.compute_loss(self.weight, self.last_output)
    
    def postprocess_parameters(self) -> None:
        with torch.no_grad():
            self.weight_postprocessor.postprocess(self.weight)
    
    def _collect_inputs(self) -> Tensor:
        inputs = [trans(module.last_output)
                  for trans, module in zip(self._transforms, self.nl_modules)]

        return torch.cat(inputs, -1)


class NLStack(NLModule):

    def __init__(self, nl_modules: List[NLModule]):
        super().__init__()

        self.nl_modules = ModuleList(nl_modules)

    def prepare_transforms(self) -> List[NLVariable]:
        for module in self.nl_modules:
            module.prepare_transforms()

        self._transforms = [lambda x, s: torch.flatten(x, 1)
                            for _ in self.nl_modules]

        return []

    def _collect_inputs(self) -> Tensor:
        inputs = [trans(module.last_output)
                  for trans, module in zip(self._transforms, self.nl_modules)]

        return torch.cat(inputs, -1)
    
    def forward(self) -> Tensor:
        self.last_output = self._collect_inputs()

        return self.last_output

    def __str__(self) -> str:
        if len(self.args) > 0:
            nl_modules = f"({', '.join(str(module) for module in self.nl_modules)})"
        else:
            nl_modules = ""

        return f"NLStack{nl_modules}"


class NLNetwork(NLModule):

    __constants__ = ["head"]

    head: NLModule

    def __init__(self, head: NLModule):
        super().__init__()

        self.head = head
        self.prepare_dependecies()
        self.prepare_transforms()

    def prepare_dependecies(self) -> None:
        dependencies = self.head.get_dependencies()
        self.nl_modules = ModuleList()

        while len(dependencies) > 0:
            removed = set()
            new_dependencies = dict()

            for module, deps in dependencies:
                if len(deps) == 0:
                    self.nl_modules.append(module)
                    removed.add(module)
                else:
                    new_dependencies[module] = deps - removed

            dependencies = new_dependencies

    def prepare_transforms(self) -> List[NLVariable]:
        return self.head.prepare_transforms()

    def reset_parameters(self) -> None:
        for module in self.nl_modules:
            if isinstance(module, NLModuleParametrized):
                module.reset_parameters()

    def compute_regularization_loss(self) -> Tensor:
        return torch.sum(module.compute_regularization_loss()
                         for module in self.nl_modules
                         if isinstance(module, NLModuleParametrized))

    def postprocess_parameters(self) -> None:
        for module in self.nl_modules:
            if isinstance(module, NLModuleParametrized):
                module.postprocess_parameters()
    
    def forward(self, structures: Iterable[Structure]) -> Tensor:
        for module in self.nl_modules:
            if isinstance(module, NLPredicateAtom):
                module.forward(structures)
            else:
                module.forward()

        self.last_output = self.head.last_output

        return self.last_output

    def __str__(self) -> str:
        head = str(self.head)

        return f"NLNetwork{head}"
