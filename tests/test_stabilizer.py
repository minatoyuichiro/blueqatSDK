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
"""Stabilizer simulation of Clifford circuits."""

import random
from collections import Counter

import pytest
import torch

from blueqat import Circuit
from blueqat.stabilizer import StabilizerSimulator

_ONE = ['h', 'x', 'y', 'z', 's', 'sdg', 'sx', 'sxdg']
_TWO = ['cx', 'cy', 'cz', 'swap']


def _random_clifford_circuit(n, depth, rng):
    circuit = Circuit(n)
    for _ in range(depth):
        if n == 1 or rng.random() < 0.5:
            getattr(circuit, rng.choice(_ONE))[rng.randrange(n)]
        else:
            a, b = rng.sample(range(n), 2)
            getattr(circuit, rng.choice(_TWO))[a, b]
    return circuit


def _support(circuit, n):
    probs = (torch.abs(circuit.run(mode='statevector')) ** 2).tolist()
    return {format(i, f'0{n}b') for i, p in enumerate(probs) if p > 1e-12}


# ----------------------------------------------------------- correctness

def test_bell_state_stabilizers():
    sim = Circuit(2).h[0].cx[0, 1].run(backend='stabilizer')
    assert sim.stabilizers() == ['+XX', '+ZZ']


def test_ghz_stabilizers():
    sim = Circuit(3).h[0].cx[0, 1].cx[1, 2].run(backend='stabilizer')
    assert sim.stabilizers() == ['+XXX', '+ZZI', '+IZZ']


def test_x_flips_the_sign_of_the_stabilizer():
    assert Circuit(1).x[0].run(backend='stabilizer').stabilizers() == ['-Z']


def test_outcomes_never_leave_the_statevector_support():
    # The sharpest cheap check: a stabilizer simulator that gets a sign wrong
    # starts producing bitstrings the true state has zero amplitude on.
    rng = random.Random(3)
    for trial in range(40):
        n = rng.randint(1, 5)
        circuit = _random_clifford_circuit(n, rng.randint(1, 12), rng)
        counts = circuit.m[:].run(backend='stabilizer', shots=200, seed=trial)
        assert set(counts) <= _support(circuit, n)


def test_distribution_matches_the_statevector_backend():
    rng = random.Random(4)
    shots = 20000
    for trial in range(6):
        n = rng.randint(1, 3)
        circuit = _random_clifford_circuit(n, rng.randint(1, 8), rng)
        exact = (torch.abs(circuit.run(mode='statevector')) ** 2).tolist()
        counts = circuit.m[:].run(backend='stabilizer', shots=shots, seed=trial)
        for index, probability in enumerate(exact):
            observed = counts.get(format(index, f'0{n}b'), 0) / shots
            assert abs(observed - probability) < 0.02


def test_deterministic_outcomes_are_exact():
    counts = Circuit(3).x[0].x[2].m[:].run(backend='stabilizer', shots=64, seed=1)
    assert counts == Counter({'101': 64})


def test_bell_pair_outcomes_are_perfectly_correlated():
    counts = Circuit(2).h[0].cx[0, 1].m[:].run(backend='stabilizer', shots=500, seed=2)
    assert set(counts) == {'00', '11'}
    assert 150 < counts['00'] < 350


# ------------------------------------------------------ measure and reset

def test_measurement_collapses_the_state():
    sim = StabilizerSimulator(1, seed=0)
    sim.apply('h', (0, ))
    first = sim.measure(0)
    assert all(sim.measure(0) == first for _ in range(10))


def test_reset_returns_to_zero():
    sim = StabilizerSimulator(2, seed=0)
    sim.apply('h', (0, ))
    sim.apply('cx', (0, 1))
    sim.reset(0)
    assert sim.measure(0) == 0


def test_reset_in_a_circuit():
    counts = Circuit(2).h[0].cx[0, 1].reset[0].m[:].run(backend='stabilizer',
                                                        shots=100, seed=5)
    # Qubit 0 is back to |0>; qubit 1 keeps whatever it collapsed to.
    assert all(key[-1] == '0' for key in counts)


def test_measuring_all_qubits_is_implied_when_no_m_is_written():
    counts = Circuit(2).h[0].cx[0, 1].run(backend='stabilizer', shots=100, seed=1)
    assert set(counts) == {'00', '11'}


def test_only_measured_qubits_are_reported():
    counts = Circuit(3).x[:].m[0].run(backend='stabilizer', shots=8, seed=1)
    assert counts == Counter({'001': 8})


# -------------------------------------------------------------- plumbing

def test_seed_makes_a_run_reproducible():
    circuit = Circuit(4).h[:].cx[0, 1].cx[2, 3]
    a = circuit.m[:].run(backend='stabilizer', shots=300, seed=7)
    assert a == circuit.m[:].run(backend='stabilizer', shots=300, seed=7)
    assert a != circuit.m[:].run(backend='stabilizer', shots=300, seed=8)


def test_bit_order_is_honored():
    counts = Circuit(3).x[0].m[:].run(backend='stabilizer', shots=8, seed=1,
                                      bit_order='q0_first')
    assert counts == Counter({'100': 8})


def test_non_clifford_gates_are_refused():
    with pytest.raises(ValueError, match='Clifford'):
        Circuit(1).t[0].run(backend='stabilizer', shots=4)
    with pytest.raises(ValueError, match='Clifford'):
        Circuit(1).rx(0.3)[0].run(backend='stabilizer', shots=4)


def test_amplitude_returns_are_refused():
    for returns in ('statevector', 'amplitude'):
        with pytest.raises(ValueError):
            Circuit(2).h[0].run(backend='stabilizer', returns=returns)


def test_copy_is_independent():
    sim = StabilizerSimulator(2, seed=0)
    sim.apply('h', (0, ))
    clone = sim.copy()
    clone.apply('x', (1, ))
    assert sim.stabilizers() != clone.stabilizers()


def test_measure_rejects_an_out_of_range_qubit():
    with pytest.raises(ValueError):
        StabilizerSimulator(2).measure(5)


# ------------------------------------------------------------------ scale

def test_runs_far_past_the_statevector_limit():
    # 200 qubits is 2**200 amplitudes; the tableau is a few thousand bits.
    n = 200
    circuit = Circuit(n).h[0]
    for q in range(n - 1):
        circuit.cx[q, q + 1]
    counts = circuit.m[:].run(backend='stabilizer', shots=20, seed=1)
    assert set(counts) <= {'0' * n, '1' * n}
    assert sum(counts.values()) == 20
