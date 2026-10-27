"""Deterministic nonlinear fixtures, independent of text assets and outcomes."""
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

from poc.runtime.torch_decoder import TinyConfig, TorchDecoder, make_tiny_decoder


@dataclass(frozen=True)
class PlantedFixture:
    teacher: TorchDecoder
    canonical_operators: tuple[nn.Module, ...]
    reuse_map: tuple[int, ...]
    permutations: tuple[torch.Tensor, ...]
    oracle: TorchDecoder
    independent_teacher: TorchDecoder


def make_planted_teacher(seed: int) -> PlantedFixture:
    """Two exact nonlinear owners hidden by independently permuted FFN neurons.

    Gate and up rows use the same permutation as down columns. Attention, norms
    and nondegenerate tied readout remain depth-specific/unchanged in the oracle.
    """
    config = TinyConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=8)
    teacher = make_tiny_decoder(config, seed=seed, dtype=torch.float64)
    independent = deepcopy(teacher)
    canonical = (deepcopy(teacher.layers[0].mlp), deepcopy(teacher.layers[1].mlp))
    mapping = (0, 1, 0, 0, 1, 1, 0, 1)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    permutations = []
    with torch.no_grad():
        for layer, owner in zip(teacher.layers, mapping, strict=True):
            permutation = torch.randperm(config.intermediate_size, generator=generator)
            permutations.append(permutation)
            operator = canonical[owner]
            layer.mlp.gate_proj.weight.copy_(operator.gate_proj.weight[permutation])
            layer.mlp.up_proj.weight.copy_(operator.up_proj.weight[permutation])
            layer.mlp.down_proj.weight.copy_(operator.down_proj.weight[:,permutation])
    oracle = deepcopy(teacher)
    for layer, owner in zip(oracle.layers, mapping, strict=True):
        layer.mlp = canonical[owner]
    return PlantedFixture(teacher, canonical, mapping, tuple(permutations), oracle, independent)


@dataclass(frozen=True)
class CausalControls:
    """Hand-checkable failures that similarity/reuse heuristics must tolerate.

    The omitted second coordinate lies in a linear readout nullspace but affects
    a nonlinear suffix; individually null substitutions interact multiplicatively.
    Approximate readout equality under a fixed tolerance is nontransitive.
    """
    base: torch.Tensor
    null_delta: torch.Tensor
    sampled: torch.Tensor
    counterexample: torch.Tensor
    nontransitive_states: tuple[torch.Tensor, ...]
    tolerance: float

    @staticmethod
    def readout(x):
        return x[..., 0]

    @staticmethod
    def suffix(x):
        return x[..., 0] + x[..., 1].square()

    @staticmethod
    def missing_coordinate(x):
        return torch.stack((x[...,0] + x[...,1].square(), x[...,1]), dim=-1)

    @staticmethod
    def nontransitive_readout(x):
        return x.sigmoid()

    @staticmethod
    def joint_suffix(first: bool, second: bool) -> float:
        return float(first) * float(second)


def make_causal_controls() -> CausalControls:
    return CausalControls(torch.tensor([1.,0.], dtype=torch.float64),
                          torch.tensor([0.,1.], dtype=torch.float64),
                          torch.tensor([[1.,0.], [2.,0.]], dtype=torch.float64),
                          torch.tensor([[0.,1.]], dtype=torch.float64),
                          tuple(torch.tensor(x, dtype=torch.float64) for x in (0., .6, 1.2)), .2)
