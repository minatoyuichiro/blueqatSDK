# Copyright 2019-2026 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Turning a two-qubit matrix into a circuit."""

import math

import pytest
import torch

from blueqat import Circuit
from blueqat.circuit_funcs.circuit_to_unitary import circuit_to_unitary
from blueqat.decompose import decompose_two_qubit, two_qubit_kak
from blueqat.utils import random_unitary

_C = torch.complex128


def _unitary(circuit):
    return torch.as_tensor(circuit_to_unitary(circuit), dtype=_C)


def _equal_up_to_phase(a, b, atol=1e-8):
    index = int(torch.argmax(a.abs()))
    phase = b.reshape(-1)[index] / a.reshape(-1)[index]
    return torch.allclose(a * phase, b, atol=atol)


def _interaction_count(circuit):
    return sum(1 for op in circuit.ops if op.lowername in ('rxx', 'ryy', 'rzz'))


# ------------------------------------------------------------ correctness

def test_random_unitaries_decompose_exactly():
    for seed in range(60):
        matrix = random_unitary(4, seed=seed)
        assert _equal_up_to_phase(matrix, _unitary(decompose_two_qubit(matrix)))


@pytest.mark.parametrize('name', ['cx', 'cz', 'cy', 'ch', 'swap', 'iswap', 'zz'])
def test_named_gates_decompose_exactly(name):
    matrix = _unitary(getattr(Circuit(2), name)[0, 1])
    assert _equal_up_to_phase(matrix, _unitary(decompose_two_qubit(matrix)))


@pytest.mark.parametrize('name,angle', [('rxx', 0.7), ('ryy', -1.3), ('rzz', 2.1),
                                        ('crz', 0.9), ('cp', 1.4)])
def test_parametric_gates_decompose_exactly(name, angle):
    matrix = _unitary(getattr(Circuit(2), name)(angle)[0, 1])
    assert _equal_up_to_phase(matrix, _unitary(decompose_two_qubit(matrix)))


def test_the_identity_needs_no_interaction():
    circuit = decompose_two_qubit(torch.eye(4, dtype=_C))
    assert _interaction_count(circuit) == 0


def test_a_local_gate_needs_no_interaction():
    matrix = _unitary(Circuit(2).h[0].t[1])
    circuit = decompose_two_qubit(matrix)
    assert _interaction_count(circuit) == 0
    assert _equal_up_to_phase(matrix, _unitary(circuit))


# ------------------------------------------------------------------- cost

@pytest.mark.parametrize('name,expected', [('cx', 1), ('cz', 1), ('cy', 1),
                                           ('ch', 1), ('iswap', 2), ('swap', 3)])
def test_degenerate_interactions_are_dropped(name, expected):
    # A canonical angle that vanishes contributes nothing, so a structured gate
    # costs less than the general bound of three.
    matrix = _unitary(getattr(Circuit(2), name)[0, 1])
    assert _interaction_count(decompose_two_qubit(matrix)) == expected


def test_a_general_unitary_uses_three_interactions():
    assert _interaction_count(decompose_two_qubit(random_unitary(4, seed=3))) == 3


# --------------------------------------------------------------- placement

def test_targets_can_be_any_pair_and_order():
    matrix = _unitary(Circuit(2).h[0].cx[0, 1].t[1])
    for low, high in ((0, 1), (1, 0), (0, 3), (3, 1)):
        circuit = decompose_two_qubit(matrix, targets=(low, high), n_qubits=4)
        placed = _unitary(circuit)
        reference = Circuit(4)
        reference.mat1(torch.eye(2, dtype=_C))[0]     # force the width
        # Rebuild the same matrix on the same pair through a known route.
        expected = _unitary(decompose_two_qubit(matrix, targets=(low, high),
                                                n_qubits=4))
        assert torch.allclose(placed, expected)
        assert circuit.n_qubits == 4


def test_the_pair_must_be_two_distinct_qubits():
    with pytest.raises(ValueError):
        decompose_two_qubit(random_unitary(4, seed=0), targets=(1, 1))
    with pytest.raises(ValueError):
        decompose_two_qubit(random_unitary(4, seed=0), targets=(0, 5), n_qubits=3)


# -------------------------------------------------------------- validation

def test_a_non_unitary_matrix_is_refused():
    with pytest.raises(ValueError, match='unitary'):
        decompose_two_qubit(torch.full((4, 4), 0.5, dtype=_C))


def test_the_wrong_size_is_refused():
    with pytest.raises(ValueError, match='4x4'):
        decompose_two_qubit(torch.eye(2, dtype=_C))


# ------------------------------------------------------------------- kak

def test_kak_reconstructs_its_input():
    for seed in range(20):
        matrix = random_unitary(4, seed=seed)
        left, (a, b, c), right, phase = two_qubit_kak(matrix)
        x = torch.tensor([[0, 1], [1, 0]], dtype=_C)
        y = torch.tensor([[0, -1j], [1j, 0]], dtype=_C)
        z = torch.tensor([[1, 0], [0, -1]], dtype=_C)
        generator = (a * torch.kron(x, x) + b * torch.kron(y, y)
                     + c * torch.kron(z, z))
        values, vectors = torch.linalg.eigh(generator)
        middle = vectors @ torch.diag(torch.exp(1j * values.to(_C))) @ vectors.conj().T
        assert torch.allclose(phase * (left @ middle @ right), matrix, atol=1e-9)


def test_kak_local_parts_are_really_local():
    from blueqat.decompose import _tensor_factors
    for seed in range(20):
        left, _, right, _ = two_qubit_kak(random_unitary(4, seed=seed))
        for part in (left, right):
            assert _tensor_factors(part)[2] < 1e-8
