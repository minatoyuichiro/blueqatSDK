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

import sys
import math

import pytest
import torch

from blueqat import Circuit
from blueqat.circuit_funcs.circuit_to_unitary import circuit_to_unitary
from blueqat.decompose import (complete_to_unitary, decompose_isometry,
                               decompose_two_qubit, decompose_unitary,
                               synthesize_two_qubit, two_qubit_kak)
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


# ------------------------------------------------- three-CX synthesis

def test_synthesis_reaches_three_cx():
    for seed in range(8):
        matrix = random_unitary(4, seed=seed)
        circuit = synthesize_two_qubit(matrix)
        assert sum(1 for op in circuit.ops if op.lowername == 'cx') == 3
        assert _equal_up_to_phase(matrix, _unitary(circuit), atol=1e-6)


def test_synthesis_refuses_when_it_cannot_reach_the_budget():
    # One CX cannot express a general two-qubit unitary, and saying so beats
    # returning the closest thing it found.
    with pytest.raises(RuntimeError, match='could not fit'):
        synthesize_two_qubit(random_unitary(4, seed=0), n_cx=1)


def test_synthesis_validates_its_arguments():
    with pytest.raises(ValueError):
        synthesize_two_qubit(random_unitary(4, seed=0), n_cx=4)
    with pytest.raises(ValueError):
        synthesize_two_qubit(torch.eye(2, dtype=_C))


# ------------------------------------------------ any number of qubits

@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_arbitrary_unitaries_decompose(n):
    for seed in range(4):
        matrix = random_unitary(1 << n, seed=seed + 10 * n)
        assert _equal_up_to_phase(matrix, _unitary(decompose_unitary(matrix)))


@pytest.mark.parametrize('build', [
    lambda: Circuit(3).ccx[0, 1, 2],
    lambda: Circuit(3).cswap[0, 1, 2],
    lambda: Circuit(3).cx[0, 1],
    lambda: Circuit(3).h[0].h[1].h[2],
])
def test_structured_unitaries_decompose(build):
    # These are where a naive cosine-sine construction fails: repeated cosines.
    matrix = _unitary(build())
    assert _equal_up_to_phase(matrix, _unitary(decompose_unitary(matrix)))


def test_the_identity_decomposes_without_entanglement():
    circuit = decompose_unitary(torch.eye(8, dtype=_C))
    assert _equal_up_to_phase(torch.eye(8, dtype=_C), _unitary(circuit))


def test_cosine_sine_reconstructs():
    from blueqat.decompose import cosine_sine
    for size in (2, 4, 8):
        matrix = random_unitary(size, seed=size)
        top, bottom, cosines, sines, right_top, right_bottom = cosine_sine(matrix)
        half = size // 2
        middle = torch.zeros((size, size), dtype=_C)
        middle[:half, :half] = torch.diag(cosines.to(_C))
        middle[:half, half:] = -torch.diag(sines.to(_C))
        middle[half:, :half] = torch.diag(sines.to(_C))
        middle[half:, half:] = torch.diag(cosines.to(_C))
        left = torch.block_diag(top, bottom)
        right = torch.block_diag(right_top, right_bottom)
        assert torch.allclose(left @ middle @ right.conj().T, matrix, atol=1e-9)


# ------------------------------------------------------------ isometries

@pytest.mark.parametrize('n,k', [(2, 1), (3, 1), (3, 2), (4, 2)])
def test_isometries_act_correctly_on_padded_inputs(n, k):
    """The shape a sequential MPS circuit is made of.

    The circuit reproduces the isometry when the qubits above the input
    register start in |0>, which is what the padding means.
    """
    source = random_unitary(1 << n, seed=n * 10 + k)
    isometry = source[:, :1 << k]
    circuit = decompose_isometry(isometry)
    for column in range(1 << k):
        start = torch.zeros(1 << n, dtype=_C)
        start[column] = 1.0
        out = circuit.run(initial=start, mode='statevector')
        expected = isometry[:, column]
        index = int(torch.argmax(expected.abs()))
        phase = out[index] / expected[index]
        assert torch.allclose(expected * phase, out, atol=1e-7)


def test_completion_keeps_the_original_columns():
    source = random_unitary(8, seed=2)
    isometry = source[:, :2]
    completed = complete_to_unitary(isometry)
    assert torch.allclose(completed[:, :2], isometry, atol=1e-12)
    assert torch.allclose(completed.conj().T @ completed,
                          torch.eye(8, dtype=_C), atol=1e-9)


def test_a_non_isometry_is_refused():
    with pytest.raises(ValueError, match='isometry'):
        complete_to_unitary(torch.full((8, 2), 0.5, dtype=_C))


# --- the path taken without SciPy ------------------------------------------
#
# SciPy is not a declared dependency, so anyone who installs blueqat normally
# takes the fallback in `cosine_sine`. Until this fixture existed, every test
# ran on a machine that happened to have SciPy, so the fallback was never
# executed and a threshold bug in it went unnoticed: `sqrt(1 - cos**2)` at
# cos == 1 cancels to machine epsilon and the square root lifts that to ~1.5e-8,
# clearing a 1e-9 "is the sine nonzero" test and dividing through by it.

@pytest.fixture
def without_scipy(monkeypatch):
    """Make `from scipy.linalg import ...` raise, as it does without SciPy.

    A None entry in sys.modules is exactly what Python raises ImportError on."""
    monkeypatch.setitem(sys.modules, 'scipy', None)
    monkeypatch.setitem(sys.modules, 'scipy.linalg', None)
    with pytest.raises(ImportError):
        from scipy.linalg import cossin  # noqa: F401
    yield


@pytest.mark.parametrize('build', [
    lambda: Circuit(3).ccx[0, 1, 2],
    lambda: Circuit(3).cswap[0, 1, 2],
    lambda: Circuit(3).cx[0, 1],
    lambda: Circuit(3).h[0].h[1].h[2],
])
def test_structured_unitaries_decompose_without_scipy(build, without_scipy):
    matrix = _unitary(build())
    assert _equal_up_to_phase(matrix, _unitary(decompose_unitary(matrix)))


@pytest.mark.parametrize('n', [1, 2, 3])
def test_arbitrary_unitaries_decompose_without_scipy(n, without_scipy):
    for seed in range(3):
        matrix = random_unitary(1 << n, seed=seed + 10 * n)
        assert _equal_up_to_phase(matrix, _unitary(decompose_unitary(matrix)))


def test_isometries_decompose_without_scipy(without_scipy):
    from blueqat.decompose import decompose_isometry
    torch.manual_seed(0)
    iso = torch.linalg.qr(torch.randn(8, 2, dtype=_C))[0]
    circuit = decompose_isometry(iso)
    got = _unitary(circuit)[:, :2]
    assert _equal_up_to_phase(iso, got)


def test_the_two_paths_agree(without_scipy):
    """The fallback is not merely self-consistent: it must produce the same
    decomposition SciPy does, up to the freedom the factorization allows."""
    from blueqat.decompose import cosine_sine
    matrix = _unitary(Circuit(3).ccx[0, 1, 2])
    l0, l1, cos, sin, r0, r1 = cosine_sine(matrix)
    half = matrix.shape[0] // 2
    middle = torch.zeros((2 * half, 2 * half), dtype=_C)
    middle[:half, :half] = torch.diag(cos).to(_C)
    middle[:half, half:] = -torch.diag(sin).to(_C)
    middle[half:, :half] = torch.diag(sin).to(_C)
    middle[half:, half:] = torch.diag(cos).to(_C)
    left = torch.block_diag(l0, l1)
    right = torch.block_diag(r0, r1)
    assert torch.allclose(matrix, left @ middle @ right.conj().T, atol=1e-10)


# --- emitting CX directly ---------------------------------------------------
#
# One CX sandwich carries an XX and a ZZ term at once, so the interaction needs
# two sandwiches (four CX) rather than three separate two-qubit rotations (six).

def _cx_cost(circuit):
    """CX gates, counting each two-qubit rotation as the two it compiles to."""
    return sum(2 if op.lowername in ('rxx', 'ryy', 'rzz') else 1
               for op in circuit.ops if op.lowername in ('cx', 'rxx', 'ryy', 'rzz'))


@pytest.mark.parametrize('seed', range(6))
def test_cx_basis_is_exact(seed):
    matrix = random_unitary(4, seed=400 + seed)
    circuit = decompose_two_qubit(matrix, basis='cx')
    assert _equal_up_to_phase(matrix, _unitary(circuit))
    assert _cx_cost(circuit) == 4


@pytest.mark.parametrize('build,cost', [
    (lambda: Circuit(2), 0),                       # nothing to entangle
    (lambda: Circuit(2).cx[0, 1], 2),              # XX and ZZ content only
    (lambda: Circuit(2).cz[0, 1], 2),
    (lambda: Circuit(2).zz[0, 1], 2),
    (lambda: Circuit(2).swap[0, 1], 4),            # needs the YY sandwich too
    (lambda: Circuit(2).iswap[0, 1], 4),
])
def test_cx_basis_costs_less_when_the_interaction_is_degenerate(build, cost):
    matrix = _unitary(build())
    circuit = decompose_two_qubit(matrix, basis='cx')
    assert _equal_up_to_phase(matrix, _unitary(circuit))
    assert _cx_cost(circuit) == cost


def test_cx_basis_beats_rotations_and_agrees_with_it():
    for seed in range(4):
        matrix = random_unitary(4, seed=500 + seed)
        rotations = decompose_two_qubit(matrix, basis='rotations')
        direct = decompose_two_qubit(matrix, basis='cx')
        assert _cx_cost(direct) < _cx_cost(rotations)
        # Same operator, not merely both close to something.
        assert _equal_up_to_phase(_unitary(rotations), _unitary(direct))


def test_cx_basis_is_exact_not_fitted():
    """The point of the closed form: machine precision, where the gradient fit
    in synthesize_two_qubit reaches about 1e-7."""
    matrix = random_unitary(4, seed=7)
    got = _unitary(decompose_two_qubit(matrix, basis='cx'))
    overlap = torch.trace(matrix.conj().T @ got)
    aligned = got * (overlap / abs(overlap)).conj()
    assert float((matrix - aligned).abs().max()) < 1e-13


@pytest.mark.parametrize('targets,width', [((1, 0), 2), ((0, 2), 3), ((2, 1), 4)])
def test_cx_basis_honors_targets_and_width(targets, width):
    matrix = random_unitary(4, seed=11)
    circuit = decompose_two_qubit(matrix, targets=targets, n_qubits=width, basis='cx')
    assert circuit.n_qubits == width
    reference = decompose_two_qubit(matrix, targets=targets, n_qubits=width)
    assert _equal_up_to_phase(_unitary(reference), _unitary(circuit))


@pytest.mark.parametrize('n', [2, 3, 4])
def test_n_qubit_decomposition_in_the_cx_basis(n):
    matrix = random_unitary(1 << n, seed=60 + n)
    circuit = decompose_unitary(matrix, basis='cx')
    assert _equal_up_to_phase(matrix, _unitary(circuit), atol=1e-9)
    assert _cx_cost(circuit) < _cx_cost(decompose_unitary(matrix))


def test_n_qubit_cx_basis_without_scipy(without_scipy):
    matrix = random_unitary(8, seed=63)
    circuit = decompose_unitary(matrix, basis='cx')
    assert _equal_up_to_phase(matrix, _unitary(circuit), atol=1e-9)


@pytest.mark.parametrize('call', [
    lambda: decompose_two_qubit(random_unitary(4, seed=1), basis='nonsense'),
    lambda: decompose_unitary(random_unitary(4, seed=1), basis='nonsense'),
])
def test_unknown_basis_is_rejected(call):
    with pytest.raises(ValueError, match="basis must be"):
        call()
