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
"""Quasi-static noise: the dephasing a Kraus channel cannot express."""

import math

import pytest
import torch

from blueqat import Circuit
from blueqat.noise import QuasiStatic, depolarizing, phase_damping


def _coherence(rho):
    """``2|rho_01|``: 1 for a fresh |+>, 0 once the phase is fully randomized."""
    return 2 * abs(complex(rho[0, 1]))


def _idle(layers, **kwargs):
    circuit = Circuit(1).h[0]
    for _ in range(layers):
        circuit.i[0]
    return circuit.run(**kwargs)


def _echo(tau, **kwargs):
    """Prepare |+>, wait, flip, wait the same again, flip back."""
    circuit = Circuit(1).h[0]
    for _ in range(tau):
        circuit.i[0]
    circuit.x[0]
    for _ in range(tau):
        circuit.i[0]
    circuit.x[0]
    return circuit.run(**kwargs)


# ----------------------------------------------------------- free decay

@pytest.mark.parametrize('idle_layers', [1, 2, 3])
def test_free_induction_decay_is_gaussian(idle_layers):
    # Averaging a static detuning over a Gaussian gives exp(-(sigma * t)**2 / 2),
    # with t counted in layers -- the h is a layer of its own, hence the +1.
    sigma = 0.4
    rho = _idle(idle_layers, quasi_static=QuasiStatic(sigma=sigma),
                samples=6000, seed=1)
    expected = math.exp(-(sigma * (idle_layers + 1)) ** 2 / 2)
    assert abs(_coherence(rho) - expected) < 0.02


def test_zero_sigma_is_noiseless():
    rho = _idle(3, quasi_static=QuasiStatic(sigma=0.0), samples=5, seed=1)
    assert abs(_coherence(rho) - 1.0) < 1e-12


# --------------------------------------------------------------- the echo

def test_a_hahn_echo_refocuses_quasi_static_noise():
    # The whole reason this is not a channel: a static offset accumulated before
    # the flip is undone after it.
    noise = dict(quasi_static=QuasiStatic(sigma=0.4), samples=3000, seed=2)
    without = _coherence(_idle(7, **noise))
    with_echo = _coherence(_echo(3, **noise))
    assert without < 0.1
    assert with_echo > 0.8


def test_a_hahn_echo_does_not_refocus_a_dephasing_channel():
    # A Markovian channel has no memory to undo, so the echo buys nothing --
    # which is exactly why reproducing a T2* experiment needs quasi_static.
    noise = dict(noise=phase_damping(0.25))
    without = _coherence(_idle(7, **noise))
    with_echo = _coherence(_echo(3, **noise))
    assert with_echo <= without + 0.05


# ------------------------------------------------------------ the average

def test_the_average_is_a_valid_density_matrix():
    rho = Circuit(2).h[0].cx[0, 1].run(quasi_static=QuasiStatic(sigma=0.3),
                                       samples=400, seed=3)
    assert abs(float(torch.diagonal(rho).sum().real) - 1.0) < 1e-10
    assert torch.allclose(rho, rho.conj().T, atol=1e-12)
    eigenvalues = torch.linalg.eigvalsh(rho).real
    assert float(eigenvalues.min()) > -1e-10


def test_results_are_reproducible_with_a_seed():
    kwargs = dict(quasi_static=QuasiStatic(sigma=0.3), samples=200)
    a = Circuit(1).h[0].i[0].run(seed=5, **kwargs)
    assert torch.allclose(a, Circuit(1).h[0].i[0].run(seed=5, **kwargs))
    assert not torch.allclose(a, Circuit(1).h[0].i[0].run(seed=6, **kwargs))


def test_it_combines_with_a_channel():
    rho = Circuit(1).h[0].i[0].run(quasi_static=QuasiStatic(sigma=0.3),
                                   noise=depolarizing(0.05), samples=300, seed=1)
    assert abs(float(torch.diagonal(rho).sum().real) - 1.0) < 1e-10
    # Both sources damp, so coherence is below either alone.
    only_static = Circuit(1).h[0].i[0].run(quasi_static=QuasiStatic(sigma=0.3),
                                           samples=300, seed=1)
    assert _coherence(rho) < _coherence(only_static)


def test_more_samples_converge_towards_the_analytic_value():
    sigma, layers = 0.5, 2
    expected = math.exp(-(sigma * (layers + 1)) ** 2 / 2)
    coarse = abs(_coherence(_idle(layers, quasi_static=QuasiStatic(sigma=sigma),
                                  samples=50, seed=9)) - expected)
    fine = abs(_coherence(_idle(layers, quasi_static=QuasiStatic(sigma=sigma),
                                samples=5000, seed=9)) - expected)
    assert fine < coarse


# ------------------------------------------------------------- scaling

def test_scaled_moves_sigma_by_the_square_root():
    # Gaussian decay goes as exp(-(sigma*t)**2/2), so the exponent -- the thing
    # extrapolation is linear in -- scales with sigma**2.
    scaled = QuasiStatic(sigma=0.2, dt=0.5).scaled(4.0)
    assert abs(scaled.sigma - 0.4) < 1e-12
    assert abs(scaled.dt - 0.5) < 1e-12


def test_noise_scale_reaches_quasi_static():
    kwargs = dict(samples=2000, seed=4)
    doubled = _idle(2, quasi_static=QuasiStatic(sigma=0.3), noise_scale=4.0, **kwargs)
    direct = _idle(2, quasi_static=QuasiStatic(sigma=0.6), **kwargs)
    assert abs(_coherence(doubled) - _coherence(direct)) < 1e-12


def test_noise_scale_still_needs_something_to_scale():
    with pytest.raises(ValueError):
        Circuit(1).h[0].run(backend='density', noise_scale=2.0)


# ---------------------------------------------------------- validation

@pytest.mark.parametrize('kwargs', [{'sigma': -0.1}, {'sigma': 0.1, 'dt': 0.0},
                                    {'sigma': 0.1, 'dt': -1.0}])
def test_bad_parameters_are_refused(kwargs):
    with pytest.raises(ValueError):
        QuasiStatic(**kwargs)


def test_negative_scale_is_refused():
    with pytest.raises(ValueError):
        QuasiStatic(0.1).scaled(-1.0)


def test_zero_samples_is_refused():
    with pytest.raises(ValueError):
        Circuit(1).h[0].run(quasi_static=QuasiStatic(0.1), samples=0)


def test_quasi_static_alone_selects_the_density_backend():
    result = Circuit(1).h[0].run(quasi_static=QuasiStatic(sigma=0.1), samples=10)
    assert result.dim() == 2


# ------------------------------------------------- leakage on a mixed state

def test_leakage_reads_a_density_matrix():
    import blueqat.eo
    from blueqat.eo import encoding

    encoded = encoding.encode_state([(1, 0)])
    pure = encoding.leakage(encoded)
    mixed = encoding.leakage(torch.outer(encoded, encoded.conj()))
    assert abs(pure - mixed) < 1e-12
    assert pure < 1e-12

    # Noise pushes population out of the encoded subspace. (The channel follows
    # each gate, so the circuit needs gates for there to be any.)
    noisy = Circuit(3).exch(0.7)[0, 1].exch(0.5)[1, 2].run(
        backend='density', initial=encoded, noise=depolarizing(0.2))
    assert encoding.leakage(noisy) > 0.01
