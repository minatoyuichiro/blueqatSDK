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
"""Ramsey, Hahn echo and CPMG, and telling two kinds of dephasing apart."""

import functools
import math

import pytest

from blueqat import Circuit
from blueqat.noise import NoiseModel, QuasiStatic, phase_damping
from blueqat.spin import (coherence_curve, coherence_of, echo_circuit,
                          fit_coherence, free_evolution_time, identify_noise,
                          ramsey_circuit, refocusing_gain)

DELAYS = [0, 2, 4, 6, 8]
SIGMA = 0.3
# A quasi-static run is a Monte Carlo average, so its cost is linear in
# `samples` and its error falls as 1/sqrt(samples). 4000 puts the error near
# 1e-2, which the tolerances below are written around; the one test that needs
# to resolve a single layer of time asks for more, on its own.
QUASI_STATIC = dict(quasi_static=QuasiStatic(SIGMA, 1.0), samples=4000, seed=1)
MARKOVIAN = dict(noise=NoiseModel().add(phase_damping(0.15)))


@functools.lru_cache(maxsize=None)
def curve(sequence, kind):
    """The same curves are wanted by several tests; computing each once takes
    this file from four minutes to well under one."""
    settings = QUASI_STATIC if kind == 'quasi_static' else MARKOVIAN
    values = coherence_curve(DELAYS, sequence, **settings)
    return tuple(values[d] for d in DELAYS)


# --- the layer-to-time conversion ------------------------------------------

@pytest.mark.parametrize('delay,expected', [(0, 1.0), (1, 2.0), (7, 8.0)])
def test_a_sequence_evolves_for_one_layer_more_than_it_waits(delay, expected):
    """The preparation layer accumulates phase too. Getting this wrong shifts a
    fitted T2* by a whole layer with nothing to show for it."""
    assert free_evolution_time(delay) == expected
    assert free_evolution_time(delay, dt=0.5) == expected * 0.5


def test_the_conversion_is_the_one_the_simulator_actually_uses():
    """Checked against the decay rather than against the docstring: at 40000
    samples the Ramsey coherence follows exp(-(sigma*T)**2/2) with T = delay+1
    to within Monte Carlo error, and does not with T = delay."""
    delay = 4                       # far enough out that the two differ clearly
    measured = coherence_of(ramsey_circuit(delay),
                            quasi_static=QuasiStatic(SIGMA, 1.0),
                            samples=40000, seed=3)
    right = math.exp(-(SIGMA * free_evolution_time(delay)) ** 2 / 2)
    wrong = math.exp(-(SIGMA * delay) ** 2 / 2)
    assert measured == pytest.approx(right, abs=5e-3)
    assert abs(measured - wrong) > 5e-2


def test_a_negative_delay_is_refused():
    with pytest.raises(ValueError, match='non-negative'):
        free_evolution_time(-1)


# --- what the sequences are ------------------------------------------------

def test_idle_time_is_spent_on_gates_not_barriers():
    """A NoiseModel attaches its channels to gates. Idling on barriers would
    leave the waiting noiseless and report an immortal qubit."""
    circuit = ramsey_circuit(4)
    assert not any(op.lowername == 'barrier' for op in circuit.ops)
    assert sum(1 for op in circuit.ops if op.lowername == 'rz') == 4


def test_idling_does_not_change_the_noiseless_answer():
    """rz(0) is exactly the identity, so a clean Ramsey closes perfectly."""
    for delay in (0, 3, 7):
        assert coherence_of(ramsey_circuit(delay), backend='density') == pytest.approx(1.0)


@pytest.mark.parametrize('n_pulses', [1, 2, 3])
def test_an_echo_puts_its_pulses_in_the_middle_of_equal_intervals(n_pulses):
    circuit = echo_circuit(12, n_pulses=n_pulses)
    kinds = [op.lowername for op in circuit.ops]
    assert kinds.count('x') == n_pulses
    assert kinds.count('rz') == 12          # the total idle time is preserved


def test_an_echo_needs_at_least_one_pulse():
    with pytest.raises(ValueError, match='at least 1'):
        echo_circuit(4, n_pulses=0)


# --- the physics that makes this worth having ------------------------------

def test_a_quasi_static_offset_decays_a_ramsey_as_a_gaussian():
    for delay, measured in zip(DELAYS, curve('ramsey', 'quasi_static')):
        time = free_evolution_time(delay)
        assert measured == pytest.approx(math.exp(-(SIGMA * time) ** 2 / 2), abs=1e-2)


def test_an_echo_refocuses_a_quasi_static_offset_completely():
    """The offset is fixed within a shot, so the phase before the pulse and the
    phase after it cancel exactly. This is the whole reason quasi-static noise
    is not a Kraus channel."""
    for measured in curve('echo', 'quasi_static'):
        assert measured == pytest.approx(1.0, abs=1e-9)


def test_an_echo_does_not_refocus_a_dephasing_channel():
    """A channel has no memory, so there is nothing to cancel. Measured, the
    echo comes out slightly *worse* -- its refocusing pulse is one more gate,
    carrying one more channel."""
    for r, e in zip(curve('ramsey', 'markovian'), curve('echo', 'markovian')):
        assert e < r
        assert e == pytest.approx(r, abs=0.1)


def test_a_dephasing_channel_decays_exponentially_with_idle_time():
    values = list(curve('ramsey', 'markovian'))
    ratios = [b / a for a, b in zip(values, values[1:])]
    assert all(r == pytest.approx(ratios[0], abs=1e-9) for r in ratios)


# --- reading the shape off the curve ---------------------------------------

def test_the_fitted_t2_star_matches_the_offset_it_was_given():
    """For exp(-(t/T)**2) against exp(-(sigma t)**2 / 2), T is sqrt(2)/sigma."""
    times = [free_evolution_time(d) for d in DELAYS]
    fit = fit_coherence(times, list(curve('ramsey', 'quasi_static')), 'gaussian')
    assert fit['T'] == pytest.approx(math.sqrt(2) / SIGMA, rel=0.05)
    assert fit['residual'] < 0.01


def test_the_shape_identifies_which_noise_it_was():
    """A single T2* is a number both explanations fit; the shape is what tells
    them apart."""
    times = [free_evolution_time(d) for d in DELAYS]

    verdict = identify_noise(times, list(curve('ramsey', 'quasi_static')))
    assert verdict['model'] == 'gaussian'
    assert verdict['ratio'] > 5          # and not by a hair

    verdict = identify_noise(times, list(curve('ramsey', 'markovian')))
    assert verdict['model'] == 'exponential'
    assert verdict['ratio'] > 5


def test_refocusing_gain_separates_them_in_one_call():
    assert refocusing_gain(DELAYS, **QUASI_STATIC)['gain'] > 0.5
    assert refocusing_gain(DELAYS, **MARKOVIAN)['gain'] < 0.0


def test_a_fit_needs_something_to_fit():
    with pytest.raises(ValueError, match='at least two points'):
        fit_coherence([1.0], [0.5])
    with pytest.raises(ValueError, match='at least two points'):
        fit_coherence([1.0, 2.0], [0.0, -0.1])     # nothing positive to log


def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="'gaussian' or 'exponential'"):
        fit_coherence([1.0, 2.0], [0.9, 0.8], 'lorentzian')


def test_a_curve_that_does_not_decay_fits_as_infinite():
    fit = fit_coherence([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], 'exponential')
    assert fit['T'] == float('inf')


# --- reading the right qubit -----------------------------------------------

def test_the_coherence_is_read_off_the_qubit_that_was_used():
    """A spectator qubit must not be mistaken for the one under test."""
    circuit = ramsey_circuit(4, qubit=1, n_qubits=3)
    assert circuit.n_qubits == 3
    assert coherence_of(circuit, qubit=1, **QUASI_STATIC) < 0.8
    # Qubit 0 was never touched, so it stays in |0> and reads as fully coherent.
    assert coherence_of(circuit, qubit=0, **QUASI_STATIC) == pytest.approx(1.0, abs=1e-9)


def test_a_state_vector_run_is_refused_with_the_reason():
    with pytest.raises(TypeError, match='density matrix'):
        coherence_of(ramsey_circuit(2))
