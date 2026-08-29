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
"""Quantum channels and the density-matrix backend."""

from collections import Counter

import pytest
import torch

from blueqat import Circuit
from blueqat.gate import IFallbackOperation, OneQubitGate, TwoQubitGate
from blueqat.noise import (NoiseModel, amplitude_damping, as_noise_model, depolarizing,
                           kraus, pauli_depolarizing, phase_damping)
from blueqat.utils import X, Y, Z, I

_I2 = torch.eye(2, dtype=torch.complex128)


# --------------------------------------------------------- brute-force reference

def _embed(op, qubits, n):
    """Full 2**n operator for a 2**k op on `qubits` (most significant first)."""
    rest = [q for q in range(n) if q not in qubits]
    full = torch.kron(op, torch.eye(1 << len(rest), dtype=torch.complex128))
    order = list(qubits) + rest
    perm_row = [order.index(n - 1 - j) for j in range(n)]
    perm = perm_row + [p + n for p in perm_row]
    return full.reshape((2,) * (2 * n)).permute(perm).reshape(1 << n, 1 << n)


def _reference(circuit, n, model=None):
    """The obvious, slow way: full matrices, U rho U-dagger, explicit Kraus sums."""
    rho = torch.zeros((1 << n, 1 << n), dtype=torch.complex128)
    rho[0, 0] = 1

    def run_op(gate, rho):
        name = gate.lowername
        if name == 'barrier':
            return rho
        if name in ('reset', 'measure'):
            # reset: |0><0| and |0><1|.  measure (unread): |0><0| and |1><1|.
            k0 = torch.zeros((2, 2), dtype=torch.complex128)
            k0[0, 0] = 1.0
            k1 = torch.zeros((2, 2), dtype=torch.complex128)
            k1[0 if name == 'reset' else 1, 1] = 1.0
            for t in gate.target_iter(n):
                ops = [_embed(k, [t], n) for k in (k0, k1)]
                rho = sum(k @ rho @ k.conj().T for k in ops)
            return rho
        if isinstance(gate, OneQubitGate):
            mat, sets = gate.matrix().to(torch.complex128), [[t] for t in gate.target_iter(n)]
        elif isinstance(gate, TwoQubitGate):
            mat = gate.matrix().to(torch.complex128)
            sets = [[t, c] for c, t in gate.control_target_iter(n)]
        elif isinstance(gate, IFallbackOperation):
            for sub in gate.fallback(n):
                rho = run_op(sub, rho)
            return rho
        else:
            raise ValueError(name)
        channels = model.channels_for(name) if model else []
        for qs in sets:
            u = _embed(mat, qs, n)
            rho = u @ rho @ u.conj().T
            for channel in channels:
                if channel.scope == 'gate':
                    ops = [_embed(k, qs, n) for k in channel.kraus(len(qs))]
                    rho = sum(k @ rho @ k.conj().T for k in ops)
                else:
                    # A per-qubit channel after a 2-qubit gate is the composition of
                    # one channel per qubit, not a single pooled Kraus set.
                    for q in qs:
                        ops = [_embed(k, [q], n) for k in channel.kraus(1)]
                        rho = sum(k @ rho @ k.conj().T for k in ops)
        return rho

    for gate in circuit.ops:
        rho = run_op(gate, rho)
    return rho


# ------------------------------------------------------------------- channels

@pytest.mark.parametrize('channel,k', [
    (depolarizing(0.1), 1), (depolarizing(0.1), 2), (pauli_depolarizing(0.1), 1),
    (amplitude_damping(0.3), 1), (phase_damping(0.2), 1),
])
def test_channels_are_trace_preserving(channel, k):
    total = sum(op.conj().T @ op for op in channel.kraus(k))
    assert torch.allclose(total, torch.eye(1 << k, dtype=torch.complex128), atol=1e-12)


@pytest.mark.parametrize('k', [1, 2])
def test_depolarizing_is_the_nielsen_chuang_form(k):
    # D_p(rho) = (1 - p) rho + p I / 2**k -- what the reproduction targets use.
    p, dim = 0.137, 1 << k
    torch.manual_seed(3)
    a = torch.randn(dim, dim, dtype=torch.complex128)
    rho = a @ a.conj().T
    rho = rho / torch.trace(rho)
    got = sum(op @ rho @ op.conj().T for op in depolarizing(p).kraus(k))
    want = (1 - p) * rho + p * torch.eye(dim, dtype=torch.complex128) / dim
    assert torch.allclose(got, want, atol=1e-12)


def test_pauli_depolarizing_relates_by_three_quarters():
    # The other convention in circulation: p as "some Pauli error occurred".
    p = 0.2
    rho = torch.tensor([[0.7, 0.3 + 0.1j], [0.3 - 0.1j, 0.3]], dtype=torch.complex128)
    a = sum(k @ rho @ k.conj().T for k in depolarizing(p).kraus(1))
    b = sum(k @ rho @ k.conj().T for k in pauli_depolarizing(3 * p / 4).kraus(1))
    assert torch.allclose(a, b, atol=1e-12)


@pytest.mark.parametrize('factory', [depolarizing, pauli_depolarizing,
                                     amplitude_damping, phase_damping])
@pytest.mark.parametrize('rate', [-0.01, 1.5])
def test_channels_reject_rates_outside_the_unit_interval(factory, rate):
    with pytest.raises(ValueError):
        factory(rate)


def test_custom_kraus_channel_must_be_trace_preserving():
    with pytest.raises(ValueError):
        kraus([torch.tensor([[1.0, 0.0], [0.0, 0.5]], dtype=torch.complex128)])


def test_custom_kraus_channel_runs():
    # A deliberate bit flip, expressed as a channel.
    flip = kraus([torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)])
    rho = Circuit(1).h[0].run(noise=flip)
    assert torch.allclose(rho, Circuit(1).h[0].x[0].run(backend='density'), atol=1e-12)


def test_custom_kraus_channel_refuses_to_be_scaled():
    flip = kraus([torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)])
    with pytest.raises(ValueError):
        Circuit(1).h[0].run(noise=flip, noise_scale=2.0)


# ------------------------------------------------------------ density backend

def test_noiseless_density_run_is_the_pure_state_projector():
    c = Circuit(3).h[0].cx[0, 1].t[2].ry(0.4)[1]
    psi = c.run(mode='statevector')
    assert torch.allclose(c.run(backend='density'), torch.outer(psi, psi.conj()), atol=1e-10)


@pytest.mark.parametrize('noise', [
    depolarizing(0.05),
    amplitude_damping(0.07),
    phase_damping(0.04),
    [depolarizing(0.03), amplitude_damping(0.02)],
])
def test_noisy_run_matches_the_brute_force_reference(noise):
    c = Circuit(3).h[0].cx[0, 1].rx(0.7)[2].cz[1, 2].t[0]
    got = c.run(noise=noise)
    want = _reference(c, 3, as_noise_model(noise))
    assert torch.allclose(got, want, atol=1e-10)


def test_two_qubit_depolarizing_acts_jointly_by_default():
    # A k-qubit depolarizing channel after a 2-qubit gate is not the same map as
    # the 1-qubit channel applied to each of its qubits.
    c = Circuit(2).h[0].cx[0, 1]
    assert not torch.allclose(c.run(noise=depolarizing(0.2)),
                              c.run(noise=depolarizing(0.2, per_qubit=True)), atol=1e-6)


def test_per_qubit_depolarizing_matches_the_reference():
    # per_qubit=True is what papers assuming purely local noise mean.
    c = Circuit(3).h[0].cx[0, 1].cz[1, 2].rx(0.3)[2]
    noise = depolarizing(0.05, per_qubit=True)
    assert torch.allclose(c.run(noise=noise), _reference(c, 3, as_noise_model(noise)),
                          atol=1e-10)


def test_per_qubit_and_joint_agree_after_one_qubit_gates():
    c = Circuit(2).h[0].t[1].rx(0.4)[0]
    assert torch.allclose(c.run(noise=depolarizing(0.1)),
                          c.run(noise=depolarizing(0.1, per_qubit=True)), atol=1e-12)


def test_per_qubit_survives_noise_scale():
    channel = depolarizing(0.01, per_qubit=True)
    scaled = channel.scaled(3.0)
    assert scaled.per_qubit and scaled.rate == pytest.approx(0.03)
    c = Circuit(2).h[0].cx[0, 1]
    assert torch.allclose(c.run(noise=channel, noise_scale=3.0),
                          c.run(noise=depolarizing(0.03, per_qubit=True)), atol=1e-12)


def test_per_qubit_depolarizing_damps_more_than_joint():
    # Two independent single-qubit channels touch more of the state than one
    # joint two-qubit channel at the same rate.
    c = Circuit(2).h[0].cx[0, 1]
    h = 1.0 * Z[0] * Z[1]
    joint = abs(float(c.run(noise=depolarizing(0.05), hamiltonian=h)))
    local = abs(float(c.run(noise=depolarizing(0.05, per_qubit=True), hamiltonian=h)))
    assert local < joint


def test_trace_is_preserved_under_noise():
    c = Circuit(3).h[:].cx[0, 1].cx[1, 2].rx(0.3)[0]
    rho = c.run(noise=[depolarizing(0.05), amplitude_damping(0.05)])
    assert abs(float(torch.diagonal(rho).sum().real) - 1.0) < 1e-10


def test_full_depolarizing_gives_the_maximally_mixed_state():
    rho = Circuit(2).h[0].cx[0, 1].run(noise=depolarizing(1.0))
    assert torch.allclose(rho, torch.eye(4, dtype=torch.complex128) / 4, atol=1e-10)


def test_full_amplitude_damping_empties_into_the_ground_state():
    rho = Circuit(1).x[0].run(noise=amplitude_damping(1.0))
    want = torch.zeros((2, 2), dtype=torch.complex128)
    want[0, 0] = 1.0
    assert torch.allclose(rho, want, atol=1e-10)


def test_phase_damping_kills_coherence_but_keeps_populations():
    rho = Circuit(1).h[0].run(noise=phase_damping(1.0))
    assert abs(float(rho[0, 0].real) - 0.5) < 1e-10
    assert abs(float(rho[1, 1].real) - 0.5) < 1e-10
    assert abs(complex(rho[0, 1])) < 1e-10


def test_noise_is_not_applied_after_measure_reset_or_barrier():
    # Only the h carries noise; the reference applies none to m/reset/barrier.
    c = Circuit(2).h[0].m[0].reset[1].barrier[:]
    assert torch.allclose(c.run(noise=depolarizing(0.1)),
                          _reference(c, 2, as_noise_model(depolarizing(0.1))), atol=1e-10)


def test_reset_returns_the_qubit_to_zero():
    rho = Circuit(1).h[0].reset[0].run(backend='density')
    want = torch.zeros((2, 2), dtype=torch.complex128)
    want[0, 0] = 1.0
    assert torch.allclose(rho, want, atol=1e-12)


def test_measure_dephases():
    rho = Circuit(1).h[0].m[0].run(backend='density')
    assert abs(complex(rho[0, 1])) < 1e-12
    assert abs(float(rho[0, 0].real) - 0.5) < 1e-12


def test_initial_accepts_a_statevector_or_a_density_matrix():
    psi = torch.tensor([0, 1], dtype=torch.complex128)
    from_vec = Circuit(1).h[0].run(backend='density', initial=psi)
    from_rho = Circuit(1).h[0].run(backend='density', initial=torch.outer(psi, psi.conj()))
    assert torch.allclose(from_vec, from_rho, atol=1e-12)


# ------------------------------------------------------------- noise_scale

def test_noise_scale_multiplies_the_rate():
    c = Circuit(2).h[0].cx[0, 1]
    assert torch.allclose(c.run(noise=depolarizing(0.02), noise_scale=3.0),
                          c.run(noise=depolarizing(0.06)), atol=1e-12)


def test_noise_scale_of_one_changes_nothing():
    c = Circuit(2).h[0].cx[0, 1]
    assert torch.allclose(c.run(noise=depolarizing(0.02), noise_scale=1.0),
                          c.run(noise=depolarizing(0.02)), atol=1e-12)


def test_noise_scale_out_of_range_is_refused_not_clipped():
    with pytest.raises(ValueError):
        Circuit(2).h[0].run(noise=depolarizing(0.4), noise_scale=5.0)


def test_noise_scale_without_noise_is_an_error():
    with pytest.raises(ValueError):
        Circuit(2).h[0].run(backend='density', noise_scale=2.0)


def test_zero_noise_extrapolation_is_a_comprehension():
    # The workflow the scale knob exists for: same circuit, several noise levels,
    # extrapolate the expectation value back to zero noise.
    c = Circuit(2).h[0].cx[0, 1]
    h = 1.0 * Z[0] * Z[1]
    exact = float(c.expect(h))
    scales = (1.0, 2.0, 3.0)
    values = [float(c.run(noise=depolarizing(0.02), noise_scale=s, hamiltonian=h))
              for s in scales]
    # Depolarizing damps <ZZ> linearly in the rate, so a linear fit lands on the
    # noise-free value.
    slope = (values[2] - values[0]) / (scales[2] - scales[0])
    extrapolated = values[0] - slope * scales[0]
    assert abs(values[0] - exact) > 1e-3          # noise really did bias it
    assert abs(extrapolated - exact) < 1e-6       # ...and extrapolation removes it


# ------------------------------------------------------------ noise models

def test_noise_model_can_give_two_qubit_gates_their_own_rate():
    model = NoiseModel()
    model.add(depolarizing(0.001))
    model.add(depolarizing(0.01), gates=['cx'])
    c = Circuit(2).h[0].cx[0, 1]
    assert torch.allclose(c.run(noise=model), _reference(c, 2, model), atol=1e-10)


def test_noise_model_gate_filter_selects():
    model = NoiseModel()
    model.add(depolarizing(0.1), gates='cx')
    assert model.channels_for('cx') and not model.channels_for('h')


def test_noise_model_scaled_leaves_the_original_alone():
    model = NoiseModel(depolarizing(0.01))
    scaled = model.scaled(2.0)
    assert model.channels_for('h')[0].rate == pytest.approx(0.01)
    assert scaled.channels_for('h')[0].rate == pytest.approx(0.02)


def test_as_noise_model_accepts_channel_list_and_model():
    assert isinstance(as_noise_model(depolarizing(0.1)), NoiseModel)
    assert isinstance(as_noise_model([depolarizing(0.1)]), NoiseModel)
    assert isinstance(as_noise_model(NoiseModel()), NoiseModel)
    with pytest.raises(TypeError):
        as_noise_model('depolarizing')


# ------------------------------------------------------- outputs from rho

def test_expectation_under_noise_matches_the_trace_form():
    h = 1.0 * Z[0] * Z[1] + 0.5 * X[0]
    c = Circuit(2).h[0].cx[0, 1]
    rho = c.run(noise=depolarizing(0.05))
    got = c.run(noise=depolarizing(0.05), hamiltonian=h)
    want = torch.trace(rho @ h.to_expr().simplify().to_matrix(2)).real
    assert abs(float(got) - float(want)) < 1e-10


def test_expectation_of_identity_under_noise_is_the_coefficient():
    assert abs(float(Circuit(2).h[0].run(noise=depolarizing(0.1), hamiltonian=2.5 * I))
               - 2.5) < 1e-10


def test_noisy_expectation_is_damped_towards_zero():
    h = 1.0 * Z[0] * Z[1]
    c = Circuit(2).h[0].cx[0, 1]
    clean = abs(float(c.expect(h)))
    noisy = abs(float(c.run(noise=depolarizing(0.1), hamiltonian=h)))
    assert noisy < clean


def test_shots_from_a_noisy_run():
    counts = Circuit(2).h[0].cx[0, 1].run(noise=depolarizing(0.05), shots=2000, seed=1)
    assert sum(counts.values()) == 2000
    # Depolarizing leaks weight into the odd-parity outcomes.
    assert set(counts) == {'00', '01', '10', '11'}
    assert counts['00'] + counts['11'] > counts['01'] + counts['10']


def test_noisy_shots_are_reproducible_with_a_seed():
    c = Circuit(2).h[0].cx[0, 1]
    a = c.run(noise=depolarizing(0.05), shots=500, seed=7)
    assert a == c.run(noise=depolarizing(0.05), shots=500, seed=7)
    assert a != c.run(noise=depolarizing(0.05), shots=500, seed=8)


def test_noisy_shots_honor_bit_order():
    counts = Circuit(3).x[0].run(noise=depolarizing(0.0), shots=16, seed=2,
                                 bit_order='q0_first')
    assert counts == Counter({'100': 16})


def test_noisy_shots_report_only_measured_qubits():
    counts = Circuit(3).x[:].m[0].run(noise=depolarizing(0.0), shots=8, seed=1)
    assert counts == Counter({'001': 8})


# ------------------------------------------------------------------ errors

def test_density_backend_refuses_statevector_returns():
    for returns in ('statevector', 'samples', 'amplitude'):
        with pytest.raises(ValueError):
            Circuit(2).h[0].run(backend='density', returns=returns)


def test_density_backend_refuses_too_many_qubits():
    with pytest.raises(MemoryError):
        Circuit(20).h[:].run(backend='density')


def test_noise_rejects_a_bad_type():
    with pytest.raises(TypeError):
        Circuit(2).h[0].run(noise=0.1)
