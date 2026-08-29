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
"""Peephole circuit optimization."""

import math
import random

import pytest
import torch

from blueqat import Circuit
from blueqat.circuit_funcs.circuit_to_unitary import circuit_to_unitary
from blueqat.circuit_funcs.flatten import flatten
from blueqat.optimize import optimize


def _unitary(circuit):
    return torch.as_tensor(circuit_to_unitary(circuit), dtype=torch.complex128)


def _names(circuit):
    return [op.lowername for op in circuit.ops]


# ------------------------------------------------------- the rewrites

def test_adjacent_self_inverses_cancel():
    assert optimize(Circuit(1).h[0].h[0]).ops == []
    assert optimize(Circuit(2).cx[0, 1].cx[0, 1]).ops == []


def test_inverse_pairs_cancel():
    assert optimize(Circuit(1).s[0].sdg[0]).ops == []
    assert optimize(Circuit(1).t[0].tdg[0]).ops == []
    assert optimize(Circuit(1).sx[0].sxdg[0]).ops == []


def test_cancellation_reaches_past_gates_on_other_qubits():
    # The two h's are adjacent *on qubit 0*, which is what matters.
    assert _names(optimize(Circuit(2).h[0].x[1].h[0])) == ['x']


def test_rotations_about_the_same_axis_merge():
    result = optimize(Circuit(2).rz(0.3)[0].x[1].rz(0.4)[0])
    assert _names(result) == ['rz', 'x']
    assert abs(float(result.ops[0].theta) - 0.7) < 1e-12


def test_merged_rotation_that_reaches_the_identity_disappears():
    assert optimize(Circuit(1).rz(math.pi)[0].rz(3 * math.pi)[0]).ops == []


def test_identity_gates_are_dropped():
    assert _names(optimize(Circuit(1).i[0].h[0].i[0])) == ['h']


def test_two_pi_rotation_is_kept_because_it_is_minus_identity():
    # rz(2*pi) == -I: dropping it would silently flip the statevector's sign.
    kept = optimize(Circuit(1).rx(2 * math.pi)[0])
    assert _names(kept) == ['rx']
    assert torch.allclose(_unitary(kept),
                          -torch.eye(2, dtype=torch.complex128), atol=1e-12)


def test_four_pi_rotation_is_dropped():
    assert optimize(Circuit(1).rx(4 * math.pi)[0]).ops == []


def test_phase_gate_period_is_two_pi():
    assert optimize(Circuit(1).p(2 * math.pi)[0]).ops == []


# --------------------------------------------------- target symmetry

def test_symmetric_gates_match_with_reversed_targets():
    assert optimize(Circuit(2).cz[0, 1].cz[1, 0]).ops == []
    merged = optimize(Circuit(2).exch(0.3)[0, 1].exch(0.4)[1, 0])
    assert _names(merged) == ['exch']
    assert abs(float(merged.ops[0].theta) - 0.7) < 1e-12


def test_asymmetric_gates_do_not_match_with_reversed_targets():
    # cx[0,1] cx[1,0] is not the identity; cancelling it would be wrong.
    result = optimize(Circuit(2).cx[0, 1].cx[1, 0])
    assert len(result.ops) == 2
    assert not torch.allclose(_unitary(result),
                              torch.eye(4, dtype=torch.complex128), atol=1e-9)


# ------------------------------------------------------------ safety

def test_barriers_block_optimization_and_survive():
    result = optimize(Circuit(1).h[0].barrier[0].h[0])
    assert _names(result) == ['h', 'barrier', 'h']


def test_measurement_blocks_optimization_and_survives():
    result = optimize(Circuit(1).h[0].m[0].h[0])
    assert _names(result) == ['h', 'measure', 'h']


def test_reset_blocks_optimization_and_survives():
    result = optimize(Circuit(1).h[0].reset[0].h[0])
    assert _names(result) == ['h', 'reset', 'h']


def test_a_trainable_zero_angle_is_not_dropped():
    # Its value is zero but its gradient is not, so removing the gate would
    # change what an optimizer sees.
    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    assert _names(optimize(Circuit(1).rx(theta)[0])) == ['rx']


def test_merging_keeps_angles_differentiable():
    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    merged = optimize(Circuit(1).rz(theta)[0].rz(theta)[0])
    assert _names(merged) == ['rz']
    from blueqat.utils import Z
    energy = merged.expect(1.0 * Z[0])
    assert energy.requires_grad


# ------------------------------------------------- correctness at large

def test_optimization_preserves_the_unitary_exactly():
    rng = random.Random(1)
    one = ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg']
    one_rot = ['rx', 'ry', 'rz', 'p']
    two = ['cx', 'cy', 'cz', 'swap']
    two_rot = ['rxx', 'ryy', 'rzz', 'cp', 'crz', 'exch']
    for _ in range(40):
        n = rng.randint(2, 3)
        circuit = Circuit(n)
        for _ in range(rng.randint(2, 12)):
            roll = rng.random()
            if roll < 0.35:
                getattr(circuit, rng.choice(one))[rng.randrange(n)]
            elif roll < 0.55:
                angle = rng.choice([0.3, math.pi, 2 * math.pi, -0.7])
                getattr(circuit, rng.choice(one_rot))(angle)[rng.randrange(n)]
            elif roll < 0.8:
                a, b = rng.sample(range(n), 2)
                getattr(circuit, rng.choice(two))[a, b]
            else:
                a, b = rng.sample(range(n), 2)
                getattr(circuit, rng.choice(two_rot))(rng.uniform(0, 6))[a, b]
        assert torch.allclose(_unitary(circuit), _unitary(optimize(circuit)), atol=1e-10)


def test_optimization_never_makes_a_circuit_longer():
    rng = random.Random(2)
    for _ in range(20):
        circuit = Circuit(3)
        for _ in range(rng.randint(2, 10)):
            if rng.random() < 0.6:
                getattr(circuit, rng.choice(['h', 'x', 's', 't']))[rng.randrange(3)]
            else:
                a, b = rng.sample(range(3), 2)
                getattr(circuit, rng.choice(['cx', 'cz']))[a, b]
        assert len(optimize(circuit).ops) <= len(flatten(circuit).ops)


def test_optimization_is_idempotent():
    circuit = Circuit(2).h[0].h[0].rz(0.3)[1].rz(0.4)[1].cx[0, 1].cx[0, 1].x[0]
    once = optimize(circuit)
    twice = optimize(once)
    assert _names(once) == _names(twice)


def test_blocks_and_slices_are_expanded():
    circuit = Circuit(3)
    with circuit.block('layer'):
        circuit.h[:]
    circuit.h[:]
    # Every h cancels its partner once blocks and slices are resolved.
    assert optimize(circuit).ops == []


# --------------------------------------------- exchange-only pulse counts

def test_optimizing_the_logical_circuit_shrinks_the_pulse_sequence():
    import blueqat.eo  # registers the 'eo' backend
    logical = Circuit(2).x[0].x[0].cx[0, 1].cx[0, 1].h[1]
    naive = logical.run(backend='eo')
    lean = optimize(logical).run(backend='eo')
    assert len(lean.ops) < len(naive.ops) / 10


def test_exchange_pulses_on_the_same_pair_merge():
    import blueqat.eo
    logical = Circuit(1).s[0].s[0].s[0].s[0]      # four pi/2 rz pulses = 2*pi
    assert optimize(logical.run(backend='eo')).ops == []
