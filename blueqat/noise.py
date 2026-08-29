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
"""Quantum channels and noise models for the density-matrix backend.

A :class:`Channel` is a completely positive trace-preserving map given by its
Kraus operators. A :class:`NoiseModel` says which channels follow which gates.
Both are consumed by ``Circuit.run(noise=...)``::

    from blueqat.noise import depolarizing
    Circuit(2).h[0].cx[0, 1].run(noise=depolarizing(0.01), shots=1000)
"""

import itertools
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch

__all__ = [
    'Channel', 'NoiseModel', 'QuasiStatic',
    'depolarizing', 'pauli_depolarizing', 'amplitude_damping', 'phase_damping',
    'kraus',
]

_I = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
_X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
_Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
_PAULIS = (_I, _X, _Y, _Z)


class Channel:
    """Base class for quantum channels.

    Subclasses provide :meth:`kraus` and :meth:`scaled`. `scope` decides how a
    channel is placed after a multi-qubit gate: ``'gate'`` applies it once to all
    of the gate's qubits jointly (what a k-qubit depolarizing channel means),
    while ``'qubit'`` applies the single-qubit channel to each of them
    independently (what damping means).
    """

    name = 'channel'
    scope = 'qubit'
    max_rate = 1.0

    def kraus(self, n_qubits: int) -> List[torch.Tensor]:
        """Kraus operators of this channel acting on `n_qubits` qubits."""
        raise NotImplementedError

    def scaled(self, factor: float) -> 'Channel':
        """The same channel with its rate multiplied by `factor`.

        This is the knob zero-noise extrapolation turns: running the same circuit
        at several `factor`s and extrapolating back to 0.
        """
        raise NotImplementedError

    def superoperator(self, n_qubits: int, dtype: torch.dtype,
                      device: torch.device) -> torch.Tensor:
        """``sum_m K_m (x) conj(K_m)``, the channel as one matrix acting on a
        vectorized density matrix's row and column wires together."""
        total: Optional[torch.Tensor] = None
        for k in self.kraus(n_qubits):
            k = k.to(dtype=dtype, device=device)
            term = torch.kron(k, k.conj())
            total = term if total is None else total + term
        assert total is not None, f"{self.name} produced no Kraus operators."
        return total

    def __repr__(self) -> str:
        return f"{self.name}({getattr(self, 'rate', '')})"


class _RateChannel(Channel):
    """A channel parameterized by a single rate in [0, max_rate]."""

    def __init__(self, rate: float) -> None:
        rate = float(rate)
        if not 0.0 <= rate <= self.max_rate:
            raise ValueError(
                f"{self.name} rate must be in [0, {self.max_rate}], got {rate}.")
        self.rate = rate

    def _extra_args(self) -> tuple:
        """Constructor arguments after `rate`, so that `scaled` reproduces the
        channel rather than dropping how it was configured."""
        return ()

    def scaled(self, factor: float) -> Channel:
        scaled_rate = self.rate * float(factor)
        if not 0.0 <= scaled_rate <= self.max_rate:
            raise ValueError(
                f"noise_scale={factor} takes {self.name}'s rate to {scaled_rate}, "
                f"outside the valid range [0, {self.max_rate}]. Lower the rate or "
                f"the scale rather than letting it be silently clipped.")
        return type(self)(scaled_rate, *self._extra_args())


class Depolarizing(_RateChannel):
    """``D_p(rho) = (1 - p) rho + p I / 2**k``: with probability `p` the state of
    the channel's `k` qubits is replaced by the maximally mixed state.

    This is the Nielsen & Chuang definition. The other convention in circulation
    reads `p` as the probability that *some* Pauli error occurred; that one is
    :func:`pauli_depolarizing`, and the two are related by
    ``p_pauli = 3 * p / 4`` on one qubit.

    After a multi-qubit gate the default is the joint ``k``-qubit channel, which
    mixes over all ``4**k`` Pauli strings at once. With ``per_qubit=True`` the
    single-qubit channel is instead applied to each of the gate's qubits
    independently -- genuinely a different map, and the one meant by papers that
    assume purely local noise. The two are equal after a one-qubit gate.
    """

    name = 'depolarizing'

    def __init__(self, rate: float, per_qubit: bool = False) -> None:
        super().__init__(rate)
        self.per_qubit = bool(per_qubit)
        self.scope = 'qubit' if self.per_qubit else 'gate'

    def _extra_args(self) -> tuple:
        return (self.per_qubit,)

    def __repr__(self) -> str:
        extra = ', per_qubit=True' if self.per_qubit else ''
        return f"depolarizing({self.rate}{extra})"

    def kraus(self, n_qubits: int) -> List[torch.Tensor]:
        dim4 = 4 ** n_qubits
        # Uniformly mixing every one of the 4**k Pauli strings (identity included)
        # is exactly a replacement by I/2**k, so the identity string keeps the
        # leftover weight and every other string gets p / 4**k.
        weight = self.rate / dim4
        identity_weight = 1.0 - self.rate + weight
        ops = []
        for i, letters in enumerate(itertools.product(range(4), repeat=n_qubits)):
            mat = _PAULIS[letters[0]]
            for letter in letters[1:]:
                mat = torch.kron(mat, _PAULIS[letter])
            amplitude = identity_weight if i == 0 else weight
            if amplitude > 0.0:
                ops.append(torch.sqrt(torch.tensor(amplitude, dtype=torch.float64)) * mat)
        return ops


class PauliDepolarizing(_RateChannel):
    """``(1 - p) rho + (p/3) (X rho X + Y rho Y + Z rho Z)``: with probability `p`
    one of the three Pauli errors occurs, uniformly. Single-qubit only."""

    name = 'pauli_depolarizing'
    scope = 'qubit'

    def kraus(self, n_qubits: int) -> List[torch.Tensor]:
        if n_qubits != 1:
            raise ValueError("pauli_depolarizing is a single-qubit channel.")
        p = self.rate
        out = [torch.sqrt(torch.tensor(1.0 - p, dtype=torch.float64)) * _I]
        for mat in (_X, _Y, _Z):
            out.append(torch.sqrt(torch.tensor(p / 3.0, dtype=torch.float64)) * mat)
        return out


class AmplitudeDamping(_RateChannel):
    """Decay of ``|1>`` towards ``|0>`` at rate `gamma` (energy relaxation, T1)."""

    name = 'amplitude_damping'
    scope = 'qubit'

    def kraus(self, n_qubits: int) -> List[torch.Tensor]:
        if n_qubits != 1:
            raise ValueError("amplitude_damping is a single-qubit channel.")
        g = self.rate
        k0 = torch.tensor([[1.0, 0.0], [0.0, (1.0 - g) ** 0.5]], dtype=torch.complex128)
        k1 = torch.tensor([[0.0, g ** 0.5], [0.0, 0.0]], dtype=torch.complex128)
        return [k0, k1]


class PhaseDamping(_RateChannel):
    """Loss of coherence without energy loss at rate `lam` (dephasing, T2)."""

    name = 'phase_damping'
    scope = 'qubit'

    def kraus(self, n_qubits: int) -> List[torch.Tensor]:
        if n_qubits != 1:
            raise ValueError("phase_damping is a single-qubit channel.")
        lam = self.rate
        k0 = torch.tensor([[1.0, 0.0], [0.0, (1.0 - lam) ** 0.5]], dtype=torch.complex128)
        k1 = torch.tensor([[0.0, 0.0], [0.0, lam ** 0.5]], dtype=torch.complex128)
        return [k0, k1]


class KrausChannel(Channel):
    """An arbitrary channel given by explicit Kraus operators.

    The operators are checked for trace preservation (``sum K† K == I``). Such a
    channel has no single rate to scale, so `noise_scale` other than 1 is refused
    rather than guessed at.
    """

    name = 'kraus'

    def __init__(self, operators: Sequence[Any], atol: float = 1e-8) -> None:
        ops = [torch.as_tensor(k, dtype=torch.complex128) for k in operators]
        if not ops:
            raise ValueError("kraus() needs at least one operator.")
        dim = ops[0].shape[0]
        for k in ops:
            if k.shape != (dim, dim):
                raise ValueError(f"Kraus operators must all be square and the same "
                                 f"size; got {tuple(k.shape)} alongside {(dim, dim)}.")
        if dim & (dim - 1) or dim < 2:
            raise ValueError(f"Kraus operator size must be a power of two, got {dim}.")
        total = sum(k.conj().T @ k for k in ops)
        if not torch.allclose(total, torch.eye(dim, dtype=torch.complex128), atol=atol):
            raise ValueError("Kraus operators are not trace preserving: "
                             "sum(K.conj().T @ K) must be the identity.")
        self.operators = ops
        self.n_qubits = dim.bit_length() - 1
        self.scope = 'gate'

    def kraus(self, n_qubits: int) -> List[torch.Tensor]:
        if n_qubits != self.n_qubits:
            raise ValueError(f"This kraus() channel acts on {self.n_qubits} qubit(s), "
                             f"but was placed after a {n_qubits}-qubit gate.")
        return list(self.operators)

    def scaled(self, factor: float) -> Channel:
        if float(factor) == 1.0:
            return self
        raise ValueError("A kraus() channel has no rate to scale; noise_scale must "
                         "be 1 for it. Express the scaling in the operators, or use "
                         "a parameterized channel.")


def depolarizing(p: float, per_qubit: bool = False) -> Depolarizing:
    """``(1 - p) rho + p I / 2**k`` -- see :class:`Depolarizing`.

    `per_qubit` switches a multi-qubit gate's noise from the joint ``k``-qubit
    channel to the single-qubit one applied to each of its qubits independently.
    """
    return Depolarizing(p, per_qubit)


def pauli_depolarizing(p: float) -> PauliDepolarizing:
    """``(1 - p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)`` -- see
    :class:`PauliDepolarizing`."""
    return PauliDepolarizing(p)


def amplitude_damping(gamma: float) -> AmplitudeDamping:
    """Energy relaxation at rate `gamma` -- see :class:`AmplitudeDamping`."""
    return AmplitudeDamping(gamma)


def phase_damping(lam: float) -> PhaseDamping:
    """Dephasing at rate `lam` -- see :class:`PhaseDamping`."""
    return PhaseDamping(lam)


def kraus(operators: Sequence[Any]) -> KrausChannel:
    """A custom channel from explicit Kraus operators -- see :class:`KrausChannel`."""
    return KrausChannel(operators)


class NoiseModel:
    """Which channels follow which gates.

    With no gate names, a channel applies after every gate; naming gates
    restricts it to those, which is how a device's larger two-qubit error rate is
    expressed::

        nm = NoiseModel()
        nm.add(depolarizing(0.001))                   # every gate
        nm.add(depolarizing(0.01), gates=['cx', 'cz'])  # ...plus more on these

    Channels added first are applied first. Measurement, reset and barrier never
    carry noise.
    """

    def __init__(self, *channels: Channel) -> None:
        self._entries: List[tuple] = []
        for channel in channels:
            self.add(channel)

    def add(self, channel: Channel,
            gates: Optional[Union[str, Iterable[str]]] = None) -> 'NoiseModel':
        """Append `channel`, optionally only after the named gates."""
        if not isinstance(channel, Channel):
            raise TypeError(f"Expected a Channel, got {type(channel).__name__}.")
        if gates is None:
            names: Optional[frozenset] = None
        elif isinstance(gates, str):
            names = frozenset([gates.lower()])
        else:
            names = frozenset(str(g).lower() for g in gates)
        self._entries.append((channel, names))
        return self

    def channels_for(self, gate_name: str) -> List[Channel]:
        """The channels that follow a gate called `gate_name`, in order."""
        gate_name = gate_name.lower()
        return [c for c, names in self._entries if names is None or gate_name in names]

    def scaled(self, factor: float) -> 'NoiseModel':
        """A copy with every channel's rate multiplied by `factor`."""
        out = NoiseModel()
        for channel, names in self._entries:
            out._entries.append((channel.scaled(factor), names))
        return out

    def is_empty(self) -> bool:
        return not self._entries

    def __repr__(self) -> str:
        parts = [repr(c) if n is None else f"{c!r} on {sorted(n)}"
                 for c, n in self._entries]
        return f"NoiseModel({', '.join(parts)})"


def as_noise_model(noise: Any) -> NoiseModel:
    """Normalize what a user passed as ``noise=`` into a :class:`NoiseModel`."""
    if isinstance(noise, NoiseModel):
        return noise
    if isinstance(noise, Channel):
        return NoiseModel(noise)
    if isinstance(noise, (list, tuple)):
        model = NoiseModel()
        for item in noise:
            if not isinstance(item, Channel):
                raise TypeError("A list passed as noise= must contain Channels; got "
                                f"{type(item).__name__}.")
            model.add(item)
        return model
    raise TypeError("noise= expects a Channel, a list of Channels, or a NoiseModel; "
                    f"got {type(noise).__name__}.")


class QuasiStatic:
    """Per-qubit frequency offsets that are fixed within a shot and redrawn
    between shots -- the dominant dephasing of silicon spin qubits.

    Nuclear (Overhauser) fields and 1/f charge noise drift far more slowly than
    a circuit runs, so each repetition sees an essentially constant detuning
    while the average over repetitions is what decoheres. **That is not a Kraus
    channel**: a channel has no memory, and the difference is not academic --
    a Hahn echo refocuses a quasi-static offset and leaves a Markovian
    dephasing channel untouched. Reproducing a T2* or an echo experiment needs
    this, not :func:`phase_damping`.

    Each shot draws an offset ``delta_q`` per qubit from ``N(0, sigma)`` and
    accumulates a phase ``rz(delta_q * dt)`` on every qubit after each layer of
    the circuit, so a refocusing pulse in the middle does what it does on
    hardware. The results are averaged as density matrices, which is exactly
    the classical mixture over the offsets.

    `sigma` is in radians of accumulated phase per unit time, and `dt` is how
    much time one circuit layer takes.

    Counting the elapsed time: a phase is accumulated after **every** layer,
    including the one that prepared the state, so a circuit of a preparation
    layer followed by `t` idle layers has an effective duration of ``t + 1``.
    Its coherence decays as ``exp(-(sigma * (t + 1) * dt)**2 / 2)``, not
    ``exp(-(sigma * t * dt)**2 / 2)``.
    """

    def __init__(self, sigma: float, dt: float = 1.0) -> None:
        sigma = float(sigma)
        if sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {sigma}.")
        if float(dt) <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}.")
        self.sigma = sigma
        self.dt = float(dt)

    def draw(self, n_qubits: int, rng: Any) -> List[float]:
        """One shot's frozen offsets, one per qubit."""
        return [rng.gauss(0.0, self.sigma) for _ in range(n_qubits)]

    def scaled(self, factor: float) -> 'QuasiStatic':
        """Scale the *noise strength* by `factor`.

        Gaussian dephasing decays as ``exp(-(sigma t)**2 / 2)``, so scaling the
        exponent -- the thing zero-noise extrapolation is linear in -- means
        scaling `sigma` by ``sqrt(factor)``, not by `factor`.
        """
        factor = float(factor)
        if factor < 0.0:
            raise ValueError(f"noise_scale must be non-negative, got {factor}.")
        return QuasiStatic(self.sigma * math.sqrt(factor), self.dt)

    def __repr__(self) -> str:
        return f"QuasiStatic(sigma={self.sigma}, dt={self.dt})"
