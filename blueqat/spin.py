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
"""Coherence experiments: Ramsey, Hahn echo and CPMG.

These are how a spin qubit's dephasing is characterized, and the reason they
are worth having in the simulator is that they *distinguish* two kinds of noise
that a single number cannot.

A quasi-static offset -- an Overhauser field or slow charge noise, fixed within
a shot and redrawn between them -- dephases a Ramsey fringe as a Gaussian and
is refocused entirely by an echo pulse. A Markovian dephasing channel decays
exponentially and is untouched by refocusing. Measuring only T2* gives one
number that both explanations fit; measuring T2* and T2 together separates
them, which is what these functions are for.

Everything here builds circuits and reads coherences. Fitting is separate
(`fit_coherence`) so that the measurement and the model stay distinguishable.

Idle time is spent on ``rz(0)`` gates rather than on barriers. Both end a
layer, so both accumulate a quasi-static phase identically, but a `NoiseModel`
attaches its channels to gates -- idling on barriers would leave the waiting
noiseless and report an immortal qubit.

The delay unit is a circuit *layer*, and one layer takes ``QuasiStatic.dt``.
`free_evolution_time` converts, and it is not the identity: the phase from a
quasi-static offset is accumulated after every layer including the one that
prepared the state, so a Ramsey sequence with `delay` idle layers has evolved
for ``delay + 1`` of them. Getting that wrong shifts a fitted T2* by a whole
layer, quietly, which is why it is a named function with a test rather than a
constant buried in an expression.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .circuit import Circuit


def free_evolution_time(delay: int, dt: float = 1.0) -> float:
    """How long a sequence with `delay` idle layers actually evolves for.

    ``(delay + 1) * dt``. The preparation layer accumulates phase too.
    """
    if delay < 0:
        raise ValueError(f"delay must be non-negative, got {delay}.")
    return (int(delay) + 1) * float(dt)


def _idle(circuit: Circuit, layers: int, qubit: int) -> Circuit:
    """`layers` layers of doing nothing, as ``rz(0)`` -- exactly the identity.

    A barrier would also end a layer, and for a quasi-static offset the two are
    indistinguishable: measured, they give the same coherence to every digit.
    They are not the same for a `NoiseModel`, which attaches channels to
    *gates*. Idling on barriers leaves idle time noiseless, so a Markovian T2
    curve comes out flat and the qubit looks immortal while it waits. Idling on
    a real gate makes waiting cost what waiting costs.
    """
    for _ in range(layers):
        circuit.rz(0.0)[qubit]
    return circuit


def ramsey_circuit(delay: int, qubit: int = 0, n_qubits: Optional[int] = None,
                   axis: str = 'x') -> Circuit:
    """Prepare a superposition, wait, and read the phase back.

    Decays at whatever rate the qubit dephases, refocusing nothing -- the T2*
    measurement. `axis` picks the readout quadrature: ``'x'`` closes with the
    same rotation that opened, ``'y'`` closes 90 degrees away, which is what
    shows a coherent detuning as a fringe rather than as decay.
    """
    width = n_qubits if n_qubits is not None else qubit + 1
    circuit = Circuit(width).h[qubit]
    _idle(circuit, delay, qubit)
    if axis == 'y':
        circuit.s[qubit]
    elif axis != 'x':
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}.")
    return circuit.h[qubit]


def echo_circuit(delay: int, n_pulses: int = 1, qubit: int = 0,
                 n_qubits: Optional[int] = None) -> Circuit:
    """Ramsey with refocusing pulses in the middle: Hahn echo, or CPMG.

    One pulse is a Hahn echo; more is CPMG, with the pulses at the centres of
    ``n_pulses`` equal intervals. A quasi-static offset accumulates the same
    phase before and after each pulse and the pulse flips the sign of what
    follows, so the two cancel and the sequence does not decay at all. Noise
    that varies within a shot is only partly refocused, and more pulses refocus
    faster components -- which is what makes CPMG a spectroscopy of the noise
    rather than just a longer-lived qubit.

    `delay` is the total idle time in layers, split as evenly as it divides.
    """
    if n_pulses < 1:
        raise ValueError(f"n_pulses must be at least 1, got {n_pulses}.")
    width = n_qubits if n_qubits is not None else qubit + 1
    circuit = Circuit(width).h[qubit]
    remaining = int(delay)
    for pulse in range(n_pulses):
        # Halves of the pulse's own interval, so the pulse sits at its centre.
        interval = remaining // (n_pulses - pulse)
        _idle(circuit, interval // 2, qubit)
        circuit.x[qubit]
        _idle(circuit, interval - interval // 2, qubit)
        remaining -= interval
    return circuit.h[qubit]


def coherence_of(circuit: Circuit, qubit: int = 0, **run_kwargs) -> float:
    """The coherence a sequence leaves, from a density-matrix run.

    ``2 * P(0) - 1`` on `qubit`: 1 when the sequence closes perfectly, 0 when
    the phase is fully randomized. Reported rather than ``P(0)`` because it is
    the quantity that decays, so a fit sees a decay to zero and not to a half.

    `run_kwargs` go to `Circuit.run`; pass `quasi_static=` or `noise=` to say
    what is dephasing it, and `samples=`/`seed=` for a quasi-static average.
    """
    rho = circuit.run(**run_kwargs)
    if not hasattr(rho, 'shape') or rho.ndim != 2:
        raise TypeError(
            "expected a density matrix. Pass noise= or quasi_static= (or "
            "backend='density'); a plain run returns a state vector.")
    n_qubits = circuit.n_qubits
    p_zero = 0.0
    for index in range(rho.shape[0]):
        if not (index >> qubit) & 1:
            p_zero += float(rho[index, index].real)
    return 2.0 * p_zero - 1.0


def coherence_curve(delays: Sequence[int], sequence: str = 'ramsey',
                    n_pulses: int = 1, qubit: int = 0,
                    n_qubits: Optional[int] = None,
                    **run_kwargs) -> Dict[int, float]:
    """`coherence_of` for each delay, keyed by delay in layers.

    `sequence` is ``'ramsey'`` or ``'echo'``. Convert the keys to times with
    `free_evolution_time` before fitting -- they are layer counts, and the
    conversion is not the identity.
    """
    if sequence not in ('ramsey', 'echo'):
        raise ValueError(f"sequence must be 'ramsey' or 'echo', got {sequence!r}.")
    out: Dict[int, float] = {}
    for delay in delays:
        if sequence == 'ramsey':
            circuit = ramsey_circuit(delay, qubit=qubit, n_qubits=n_qubits)
        else:
            circuit = echo_circuit(delay, n_pulses=n_pulses, qubit=qubit,
                                   n_qubits=n_qubits)
        out[int(delay)] = coherence_of(circuit, qubit=qubit, **run_kwargs)
    return out


def fit_coherence(times: Sequence[float], coherences: Sequence[float],
                  model: str = 'gaussian') -> Dict[str, float]:
    """Fit a decay and say how well it fitted.

    ``'gaussian'`` fits ``exp(-(t/T)**2)``, the shape a quasi-static offset
    gives; ``'exponential'`` fits ``exp(-t/T)``, the shape a memoryless channel
    gives. Both are linear in ``log C`` against ``t**2`` or ``t``, so this is a
    least-squares line through the origin rather than an optimizer.

    Returns ``{'T': ..., 'residual': ...}``. The residual is the root mean
    square difference between the fitted and measured coherences, and it is the
    point: fitting one model always yields a number, and the number means
    nothing until compared against the other model's. `identify_noise` does
    that comparison.

    Non-positive coherences are dropped -- the logarithm has nothing to say
    about them, and at that point the sequence has decayed anyway.
    """
    if model not in ('gaussian', 'exponential'):
        raise ValueError(f"model must be 'gaussian' or 'exponential', got {model!r}.")
    pairs = [(float(t), float(c)) for t, c in zip(times, coherences) if c > 1e-12]
    if len(pairs) < 2:
        raise ValueError("need at least two points with positive coherence to fit.")
    # log C = -(t/T)**2  or  -t/T, both of the form log C = -k * x.
    xs = [t * t if model == 'gaussian' else t for t, _ in pairs]
    ys = [-math.log(c) for _, c in pairs]
    denominator = sum(x * x for x in xs)
    if denominator <= 0.0:
        raise ValueError("all times are zero; nothing to fit.")
    k = sum(x * y for x, y in zip(xs, ys)) / denominator
    if k <= 0.0:
        return {'T': float('inf'), 'residual': _rms(
            [c for _, c in pairs], [1.0] * len(pairs))}
    T = (1.0 / math.sqrt(k)) if model == 'gaussian' else (1.0 / k)
    fitted = [math.exp(-k * x) for x in xs]
    return {'T': T, 'residual': _rms([c for _, c in pairs], fitted)}


def _rms(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def identify_noise(times: Sequence[float],
                   coherences: Sequence[float]) -> Dict[str, object]:
    """Which decay shape the data prefers, and by how much.

    Returns both fits and the name of the better one. This is the whole reason
    to measure a curve rather than a single number: T2* alone is a number that
    a quasi-static offset and a dephasing channel both explain, and the shape
    is what tells them apart -- Gaussian for the offset, exponential for the
    channel.

    ⚠ The verdict is only as good as the separation. Read `ratio`: near 1 the
    data does not choose, which happens on short curves and on noisy ones, and
    reporting a winner then is reporting a coin flip.
    """
    gaussian = fit_coherence(times, coherences, 'gaussian')
    exponential = fit_coherence(times, coherences, 'exponential')
    better = 'gaussian' if gaussian['residual'] <= exponential['residual'] else 'exponential'
    worse = exponential if better == 'gaussian' else gaussian
    best = gaussian if better == 'gaussian' else exponential
    ratio = (worse['residual'] / best['residual']) if best['residual'] > 1e-15 else float('inf')
    return {'model': better, 'gaussian': gaussian, 'exponential': exponential,
            'ratio': ratio}


def refocusing_gain(delays: Sequence[int], qubit: int = 0,
                    n_qubits: Optional[int] = None, n_pulses: int = 1,
                    **run_kwargs) -> Dict[str, float]:
    """How much an echo recovers, which is the measurement that separates the
    two noise types in one call.

    Runs the same delays as Ramsey and as echo and reports the mean coherence
    of each. A quasi-static offset is refocused completely, so `echo` stays at
    1 while `ramsey` decays; a dephasing channel is not refocused at all, so
    the two agree. `gain` is the difference, and it is the discriminator: near
    zero means refocusing bought nothing, and no amount of echo will help.
    """
    ramsey = coherence_curve(delays, 'ramsey', qubit=qubit, n_qubits=n_qubits,
                             **run_kwargs)
    echo = coherence_curve(delays, 'echo', n_pulses=n_pulses, qubit=qubit,
                           n_qubits=n_qubits, **run_kwargs)
    mean_ramsey = sum(ramsey.values()) / len(ramsey)
    mean_echo = sum(echo.values()) / len(echo)
    return {'ramsey': mean_ramsey, 'echo': mean_echo,
            'gain': mean_echo - mean_ramsey}
