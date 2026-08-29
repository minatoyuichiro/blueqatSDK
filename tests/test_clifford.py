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
"""Clifford tableaus: sampling, composition, inverse and synthesis."""

import collections
import random

import pytest
import torch

from blueqat import Circuit
from blueqat.circuit_funcs.circuit_to_unitary import circuit_to_unitary
from blueqat.clifford import Clifford, random_clifford

_PAULI = {
    'I': torch.eye(2, dtype=torch.complex128),
    'X': torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    'Y': torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
    'Z': torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
}


def _embed(letters, n):
    """Kron of single-qubit Paulis; qubit 0 is the least significant bit."""
    out = None
    for q in range(n - 1, -1, -1):
        mat = _PAULI[letters[q]]
        out = mat if out is None else torch.kron(out, mat)
    return out


def _row_matrix(clifford, row, n):
    letters = []
    for q in range(n):
        xq, zq = (clifford.x[row] >> q) & 1, (clifford.z[row] >> q) & 1
        letters.append('Y' if xq and zq else 'IXZ'[xq + 2 * zq])
    mat = _embed(letters, n)
    return -mat if clifford.phase[row] else mat


def _tableau_agrees_with_unitary(circuit, n):
    """The tableau must really be U P U-dagger for each generator P."""
    u = torch.as_tensor(circuit_to_unitary(circuit), dtype=torch.complex128)
    clifford = Clifford.from_circuit(circuit, n)
    for q in range(n):
        for row, letter in ((q, 'X'), (n + q, 'Z')):
            generator = _embed(['I'] * q + [letter] + ['I'] * (n - q - 1), n)
            if not torch.allclose(u @ generator @ u.conj().T,
                                  _row_matrix(clifford, row, n), atol=1e-9):
                return False
    return True


# ------------------------------------------------------------------ tableau

def test_identity_tableau():
    identity = Clifford.identity(2)
    assert identity == Clifford.from_circuit(Circuit(2), 2)
    assert identity.to_circuit().ops == []


@pytest.mark.parametrize('build', [
    lambda: Circuit(1).h[0],
    lambda: Circuit(1).s[0],
    lambda: Circuit(1).sdg[0],
    lambda: Circuit(1).sx[0],
    lambda: Circuit(1).sxdg[0],
    lambda: Circuit(1).x[0].y[0].z[0],
    lambda: Circuit(2).h[0].cx[0, 1],
    lambda: Circuit(2).h[0].cz[0, 1].s[1],
    lambda: Circuit(2).cy[0, 1].h[1],
    lambda: Circuit(2).swap[0, 1].h[0],
    lambda: Circuit(3).h[0].cx[0, 1].cx[1, 2].s[2].sdg[0],
])
def test_from_circuit_matches_real_conjugation(build):
    circuit = build()
    assert _tableau_agrees_with_unitary(circuit, circuit.n_qubits)


def test_from_circuit_matches_real_conjugation_on_random_circuits():
    rng = random.Random(0)
    one = ['h', 'x', 'y', 'z', 's', 'sdg', 'sx', 'sxdg']
    two = ['cx', 'cy', 'cz', 'swap']
    for _ in range(25):
        n = rng.randint(1, 3)
        circuit = Circuit(n)
        for _ in range(rng.randint(1, 8)):
            if n == 1 or rng.random() < 0.55:
                getattr(circuit, rng.choice(one))[rng.randrange(n)]
            else:
                a, b = rng.sample(range(n), 2)
                getattr(circuit, rng.choice(two))[a, b]
        assert _tableau_agrees_with_unitary(circuit, n)


def test_non_clifford_gates_are_rejected():
    for build in (lambda: Circuit(1).t[0], lambda: Circuit(1).rx(0.3)[0],
                  lambda: Circuit(2).ccx[0, 1, 0]):
        with pytest.raises(ValueError):
            Clifford.from_circuit(build())


def test_measurement_is_rejected():
    with pytest.raises(ValueError):
        Clifford.from_circuit(Circuit(1).h[0].m[0])


# ---------------------------------------------------------------- synthesis

@pytest.mark.parametrize('n', [1, 2, 3])
def test_to_circuit_round_trips(n):
    for seed in range(12):
        clifford = random_clifford(n, seed=seed)
        assert Clifford.from_circuit(clifford.to_circuit(), n) == clifford


def test_synthesized_circuit_uses_only_clifford_primitives():
    circuit = random_clifford(3, seed=5).to_circuit()
    assert set(g.lowername for g in circuit.ops) <= {'h', 's', 'sdg', 'cx', 'x', 'z'}


def test_to_circuit_reproduces_the_unitary_up_to_phase():
    original = Circuit(2).h[0].cx[0, 1].s[1]
    rebuilt = Clifford.from_circuit(original, 2).to_circuit()
    assert _tableau_agrees_with_unitary(rebuilt, 2)
    assert Clifford.from_circuit(rebuilt, 2) == Clifford.from_circuit(original, 2)


# ------------------------------------------------------------------ algebra

@pytest.mark.parametrize('n', [1, 2, 3])
def test_then_is_circuit_concatenation(n):
    for seed in range(12):
        a, b = random_clifford(n, seed=seed), random_clifford(n, seed=100 + seed)
        assert a.then(b) == Clifford.from_circuit(a.to_circuit() + b.to_circuit(), n)


@pytest.mark.parametrize('n', [1, 2, 3])
def test_inverse_cancels_on_both_sides(n):
    for seed in range(12):
        clifford = random_clifford(n, seed=seed)
        assert clifford.then(clifford.inverse()) == Clifford.identity(n)
        assert clifford.inverse().then(clifford) == Clifford.identity(n)


def test_then_is_ordered_first_then_second():
    a = Clifford.from_circuit(Circuit(1).h[0])
    b = Clifford.from_circuit(Circuit(1).s[0])
    assert a.then(b) == Clifford.from_circuit(Circuit(1).h[0].s[0])
    assert a.then(b) != Clifford.from_circuit(Circuit(1).s[0].h[0])


def test_then_rejects_a_width_mismatch():
    with pytest.raises(ValueError):
        random_clifford(1, seed=0).then(random_clifford(2, seed=0))


# ------------------------------------------------------------------ sampling

def test_random_clifford_is_reproducible():
    assert random_clifford(3, seed=11) == random_clifford(3, seed=11)
    assert random_clifford(3, seed=11) != random_clifford(3, seed=12)


def test_random_clifford_does_not_disturb_the_global_rng():
    random.seed(4)
    expected = [random.random() for _ in range(3)]
    random.seed(4)
    random_clifford(3, seed=99)
    assert [random.random() for _ in range(3)] == expected


def test_random_clifford_covers_the_whole_one_qubit_group():
    # |C_1| = 24 modulo global phase.
    seen = collections.Counter(random_clifford(1, seed=s) for s in range(2000))
    assert len(seen) == 24
    # Uniform to within a wide band: 2000 draws over 24 bins, mean 83.
    assert min(seen.values()) > 40


def test_random_clifford_two_qubit_group_is_large_and_valid():
    # |C_2| = 11520. Drawing 3000 times from 11520 equally likely elements leaves
    # 11520 * (1 - (1 - 1/11520)**3000) ~= 2640 distinct ones; a sampler biased
    # towards a subgroup would land well below that band, and one that could not
    # reach parts of the group well above it is impossible.
    seen = {random_clifford(2, seed=s) for s in range(3000)}
    assert 2500 < len(seen) < 2800
    for seed in (0, 1, 2, 3):
        clifford = random_clifford(2, seed=seed)
        assert Clifford.from_circuit(clifford.to_circuit(), 2) == clifford


def test_random_clifford_rejects_zero_qubits():
    with pytest.raises(ValueError):
        random_clifford(0)


# ------------------------------------------- randomized benchmarking use case

def _rb_circuit(n, m, seed):
    """A sequence of m random Cliffords plus the single recovery Clifford."""
    total = Clifford.identity(n)
    circuit = Circuit(n)
    for i in range(m):
        clifford = random_clifford(n, seed=seed * 1000 + i)
        circuit += clifford.to_circuit()
        total = total.then(clifford)
    return circuit + total.inverse().to_circuit()


@pytest.mark.parametrize('n', [1, 2, 3])
@pytest.mark.parametrize('m', [1, 2, 5, 12])
def test_noiseless_benchmarking_sequence_returns_to_zero(n, m):
    # The whole point of the recovery Clifford: with no noise, survival is 1.
    state = _rb_circuit(n, m, seed=m).run(mode='statevector')
    assert abs(float(abs(state[0]) ** 2) - 1.0) < 1e-9


def test_benchmarking_survival_decays_under_noise():
    from blueqat.noise import depolarizing
    short = _rb_circuit(1, 2, seed=2).run(noise=depolarizing(0.02))
    long = _rb_circuit(1, 24, seed=24).run(noise=depolarizing(0.02))
    assert float(long[0, 0].real) < float(short[0, 0].real)


def test_recovery_is_one_clifford_not_a_replay_of_the_sequence():
    # Inverting through the tableau keeps the recovery short; replaying the
    # sequence backwards would roughly double the circuit.
    n, m = 2, 20
    total = Clifford.identity(n)
    forward = Circuit(n)
    for i in range(m):
        clifford = random_clifford(n, seed=i)
        forward += clifford.to_circuit()
        total = total.then(clifford)
    recovery = total.inverse().to_circuit()
    assert len(recovery.ops) < len(forward.ops) / 2
