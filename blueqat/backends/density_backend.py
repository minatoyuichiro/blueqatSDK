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
"""Density-matrix backend: circuits with noise.

The state is a density matrix ``rho`` held as a vectorized ``2**(2n)`` tensor,
with `n` "row" wires and `n` "column" wires. That representation is what makes
noise cheap: ``rho -> U rho U†`` is ``U`` on the row wires and ``conj(U)`` on the
column wires, and a Kraus channel ``sum_m K_m rho K_m†`` is the single matrix
``sum_m K_m (x) conj(K_m)`` on those same wires. A gate and the channels that
follow it therefore multiply into **one** operator, applied in a single pass --
about eight times faster than applying each Kraus operator separately.

Cost is ``O(4**n)`` per gate, so this backend is for small circuits: comfortable
to about 10 qubits, usable to 12.
"""

from collections import Counter
import typing
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..gate import (IFallbackOperation, Operation, OneQubitGate, TwoQubitGate)
from .backendbase import Backend, BIT_ORDERS, apply_bit_order

DEFAULT_SHOTS: int = 1024

# Beyond this the density matrix alone runs into tens of gigabytes; refuse with an
# explanation rather than letting the allocator fail.
MAX_QUBITS: int = 14


def _reset_kraus(dtype: torch.dtype, device: torch.device) -> List[torch.Tensor]:
    """``|0><0|`` and ``|0><1|``: collapse and, if it landed on |1>, flip back."""
    k0 = torch.zeros((2, 2), dtype=dtype, device=device)
    k0[0, 0] = 1.0
    k1 = torch.zeros((2, 2), dtype=dtype, device=device)
    k1[0, 1] = 1.0
    return [k0, k1]


def _measure_kraus(dtype: torch.dtype, device: torch.device) -> List[torch.Tensor]:
    """``|0><0|`` and ``|1><1|``: an unread measurement is exactly dephasing."""
    k0 = torch.zeros((2, 2), dtype=dtype, device=device)
    k0[0, 0] = 1.0
    k1 = torch.zeros((2, 2), dtype=dtype, device=device)
    k1[1, 1] = 1.0
    return [k0, k1]


def _interleave_layer_phases(gates: List[Operation], n_qubits: int,
                             offsets: Sequence[float], dt: float) -> List[Operation]:
    """Insert ``rz(offset_q * dt)`` on every qubit after each layer of `gates`.

    A layer ends when an operation would reuse a qubit already busy in it (or at
    a barrier), which is the ASAP schedule ``Circuit.depth()`` counts. Applying
    the phase per layer rather than once at the end is what lets a refocusing
    pulse in the middle of the circuit actually refocus.
    """
    from ..gate import RZGate

    def phase_layer() -> List[Operation]:
        return [RZGate((q, ), offsets[q] * dt)
                for q in range(n_qubits) if offsets[q] != 0.0]

    out: List[Operation] = []
    busy: set = set()
    for gate in gates:
        qubits = set(gate.target_iter(n_qubits))
        if gate.lowername == 'barrier' or (busy & qubits):
            out.extend(phase_layer())
            busy = set()
        out.append(gate)
        busy |= qubits
    if busy:
        out.extend(phase_layer())
    return out


def _superoperator(kraus: Sequence[torch.Tensor]) -> torch.Tensor:
    total: Optional[torch.Tensor] = None
    for k in kraus:
        term = torch.kron(k, k.conj())
        total = term if total is None else total + term
    assert total is not None
    return total


class DensityMatrixBackend(Backend):
    """Runs a circuit as a density matrix, optionally with noise after each gate.

    Reached as ``Circuit.run(backend='density')``, or simply by passing
    ``noise=`` to any of ``Circuit``'s run entry points, which route here
    automatically.
    """

    #: What tells the caller that a plain run returns a density matrix rather
    #: than a statevector, so that e.g. `Circuit.probs` reads the diagonal.
    returns_density_matrix = True

    def run(self, gates: List[Operation], n_qubits: int, *args: Any, **kwargs: Any) -> Any:
        from ..noise import as_noise_model

        n_qubits = max(n_qubits, 1)
        if n_qubits > MAX_QUBITS:
            raise MemoryError(
                f"The density matrix for {n_qubits} qubits has 4**{n_qubits} entries "
                f"(the backend refuses above {MAX_QUBITS}). Noise simulation here is "
                f"meant for small circuits; drop qubits or run without noise.")

        dtype = kwargs.get("dtype", torch.complex128)
        device = kwargs.get("device", torch.device("cpu"))
        shots = kwargs.get("shots")
        returns = kwargs.get("returns")
        hamiltonian = kwargs.get("hamiltonian")
        seed = kwargs.get("seed")
        bit_order = kwargs.get("bit_order", "q0_last")
        if bit_order not in BIT_ORDERS:
            raise ValueError(f"bit_order must be one of {BIT_ORDERS}, got {bit_order!r}.")
        if returns in ("samples", "statevector", "statevector_and_shots", "amplitude"):
            raise ValueError(
                f"returns={returns!r} is a statevector notion; the density-matrix "
                f"backend returns a density matrix, shots, or an expectation value.")

        noise = kwargs.get("noise")
        model = as_noise_model(noise) if noise is not None else None
        quasi_static = kwargs.get("quasi_static")
        noise_scale = kwargs.get("noise_scale", 1.0)
        if float(noise_scale) != 1.0:
            if model is None and quasi_static is None:
                raise ValueError("noise_scale= was given without noise= or "
                                 "quasi_static=; there is nothing to scale.")
            if model is not None:
                model = model.scaled(float(noise_scale))
            if quasi_static is not None:
                quasi_static = quasi_static.scaled(float(noise_scale))
        samples = int(kwargs.get("samples", 200))

        initial = kwargs.get("initial")
        if quasi_static is None:
            state = self._initial_state(initial, n_qubits, dtype, device)
            state = self._run_gates(state, gates, n_qubits, model, dtype, device)
            rho = state.reshape(1 << n_qubits, 1 << n_qubits)
        else:
            rho = self._quasi_static_average(gates, n_qubits, model, quasi_static,
                                             samples, seed, initial, dtype, device)

        if hamiltonian is not None:
            from ..utils import pauli_expectation
            return pauli_expectation(hamiltonian, rho, n_qubits)

        if shots is None and returns != "shots":
            return rho

        from .torch_backend import has_nonterminal_measurement
        if has_nonterminal_measurement(gates, n_qubits):
            # A measured qubit is used again, so the reported bit has to be the
            # one the measurement actually produced. Averaging that away into the
            # final diagonal -- which is what sampling at the end does -- reports
            # whatever later gates left behind instead.
            return self._sample_with_collapse(gates, n_qubits, model, quasi_static,
                                              samples, shots, seed, initial, dtype,
                                              device, bit_order)
        return self._sample(rho, gates, n_qubits, shots, seed, bit_order)

    def _quasi_static_average(self, gates: List[Operation], n_qubits: int, model: Any,
                              quasi_static: Any, samples: int, seed: Optional[int],
                              initial: Any, dtype: torch.dtype,
                              device: torch.device) -> torch.Tensor:
        """Average the density matrix over frozen per-qubit detunings.

        Each sample holds its offsets fixed for the whole circuit -- that time
        correlation is the entire point, and it is why an echo sequence
        refocuses this noise while a dephasing channel survives one.
        """
        import random as _random

        if samples < 1:
            raise ValueError(f"samples must be at least 1, got {samples}.")
        rng = _random.Random(seed)
        dim = 1 << n_qubits
        total: Optional[torch.Tensor] = None
        for _ in range(samples):
            offsets = quasi_static.draw(n_qubits, rng)
            ops = _interleave_layer_phases(gates, n_qubits, offsets, quasi_static.dt)
            state = self._initial_state(initial, n_qubits, dtype, device)
            state = self._run_gates(state, ops, n_qubits, model, dtype, device)
            rho = state.reshape(dim, dim)
            total = rho if total is None else total + rho
        assert total is not None
        return total / samples

    # ------------------------------------------------------------------ state

    def _initial_state(self, initial: Any, n_qubits: int, dtype: torch.dtype,
                       device: torch.device) -> torch.Tensor:
        dim = 1 << n_qubits
        if initial is None:
            rho = torch.zeros((dim, dim), dtype=dtype, device=device)
            rho[0, 0] = 1.0
        else:
            initial = torch.as_tensor(initial, dtype=dtype, device=device)
            if initial.dim() == 1:
                # A pure state given as a statevector.
                if initial.shape[0] != dim:
                    raise ValueError(f"initial statevector must have {dim} entries, "
                                     f"got {initial.shape[0]}.")
                rho = torch.outer(initial, initial.conj())
            elif initial.dim() == 2:
                if initial.shape != (dim, dim):
                    raise ValueError(f"initial density matrix must be {dim}x{dim}, "
                                     f"got {tuple(initial.shape)}.")
                rho = initial.clone()
            else:
                raise ValueError("initial must be a statevector or a density matrix.")
        return rho.reshape((2,) * (2 * n_qubits))

    # ------------------------------------------------------------------ gates

    def _run_gates(self, state: torch.Tensor, gates: List[Operation], n_qubits: int,
                   model: Any, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        for gate in gates:
            state = self._apply_operation(state, gate, n_qubits, model, dtype, device)
        return state

    def _apply_operation(self, state: torch.Tensor, gate: Operation, n_qubits: int,
                         model: Any, dtype: torch.dtype,
                         device: torch.device) -> torch.Tensor:
        name = gate.lowername

        if name == 'barrier':
            return state
        if name == 'reset':
            sup = _superoperator(_reset_kraus(dtype, device))
            for t in gate.target_iter(n_qubits):
                state = self._apply(state, sup, self._wires([t], n_qubits), n_qubits)
            return state
        if name == 'measure':
            sup = _superoperator(_measure_kraus(dtype, device))
            for t in gate.target_iter(n_qubits):
                state = self._apply(state, sup, self._wires([t], n_qubits), n_qubits)
            return state

        # `qubit_sets` lists, per application of this gate, the qubits it acts on in
        # the order its own matrix() indexes them (most significant first).
        if isinstance(gate, OneQubitGate):
            matrix = gate.matrix().to(dtype=dtype, device=device)
            qubit_sets = [[t] for t in gate.target_iter(n_qubits)]
        elif isinstance(gate, TwoQubitGate):
            # TwoQubitGate.matrix() puts the target in the more significant bit
            # (row/col = target*2 + control).
            matrix = gate.matrix().to(dtype=dtype, device=device)
            qubit_sets = [[t, c] for c, t in gate.control_target_iter(n_qubits)]
        elif isinstance(gate, IFallbackOperation):
            for sub in gate.fallback(n_qubits):
                state = self._apply_operation(state, sub, n_qubits, model, dtype, device)
            return state
        else:
            raise ValueError(f"Cannot run {name} on the density-matrix backend.")

        channels = model.channels_for(name) if model is not None else []
        for qubits in qubit_sets:
            for wires, sup in self._fused_ops(matrix, qubits, channels, n_qubits,
                                              dtype, device):
                state = self._apply(state, sup, wires, n_qubits)
        return state

    def _fused_ops(self, matrix: torch.Tensor, qubits: Sequence[int], channels: Sequence[Any],
                   n_qubits: int, dtype: torch.dtype,
                   device: torch.device) -> List[Tuple[Tuple[int, ...], torch.Tensor]]:
        """The gate and its trailing channels as (wires, superoperator) pairs, with
        neighbours acting on identical wires multiplied together so they cost one
        pass instead of several."""
        ops: List[Tuple[Tuple[int, ...], torch.Tensor]] = [
            (self._wires(qubits, n_qubits), torch.kron(matrix, matrix.conj()))
        ]
        for channel in channels:
            if channel.scope == 'gate':
                sup = channel.superoperator(len(qubits), dtype, device)
                ops.append((self._wires(qubits, n_qubits), sup))
            else:
                sup = channel.superoperator(1, dtype, device)
                for q in qubits:
                    ops.append((self._wires([q], n_qubits), sup))

        fused: List[Tuple[Tuple[int, ...], torch.Tensor]] = []
        for wires, sup in ops:
            if fused and fused[-1][0] == wires:
                fused[-1] = (wires, sup @ fused[-1][1])
            else:
                fused.append((wires, sup))
        return fused

    @staticmethod
    def _wires(qubits: Sequence[int], n_qubits: int) -> Tuple[int, ...]:
        """Axis numbers of `qubits` in the vectorized density matrix: the row copy
        first, then the column copy, matching ``kron(U, conj(U))``.

        Reshaping a ``(2**n, 2**n)`` matrix to ``(2,) * 2n`` makes axis j the
        *most* significant bit of the row index, i.e. qubit ``n-1-j``.
        """
        rows = tuple(n_qubits - 1 - q for q in qubits)
        cols = tuple(2 * n_qubits - 1 - q for q in qubits)
        return rows + cols

    @staticmethod
    def _apply(state: torch.Tensor, op: torch.Tensor, wires: Tuple[int, ...],
               n_wires_half: int) -> torch.Tensor:
        """Contract `op` (a ``2**k x 2**k`` matrix) into `state` along `wires`."""
        n_wires = 2 * n_wires_half
        k = len(wires)
        op = op.reshape((2,) * (2 * k))
        state = torch.tensordot(op, state, dims=(list(range(k, 2 * k)), list(wires)))
        # tensordot leaves the k contracted axes at the front, in `wires` order, and
        # the untouched axes after them in their original relative order.
        rest = [w for w in range(n_wires) if w not in wires]
        current = list(wires) + rest
        position = {axis: i for i, axis in enumerate(current)}
        return state.permute([position[w] for w in range(n_wires)])

    def _sample_with_collapse(self, gates: List[Operation], n_qubits: int, model: Any,
                              quasi_static: Any, samples: int, shots: Optional[int],
                              seed: Optional[int], initial: Any, dtype: torch.dtype,
                              device: torch.device, bit_order: str) -> 'typing.Counter[str]':
        """Shot by shot, collapsing at each measurement and recording what it gave.

        Quasi-static offsets are drawn per shot here as well, so a frozen
        detuning still lasts exactly one repetition.
        """
        import random as _random

        n_shots = shots if shots is not None else DEFAULT_SHOTS
        rng = _random.Random(seed)
        measured = sorted({q for g in gates if g.lowername == 'measure'
                           for q in g.target_iter(n_qubits)})
        report = measured if measured else list(range(n_qubits))
        counts: 'typing.Counter[str]' = Counter()

        for _ in range(n_shots):
            ops = gates
            if quasi_static is not None:
                offsets = quasi_static.draw(n_qubits, rng)
                ops = _interleave_layer_phases(gates, n_qubits, offsets, quasi_static.dt)
            state = self._initial_state(initial, n_qubits, dtype, device)
            results: Dict[int, int] = {}
            for gate in ops:
                if gate.lowername == 'measure':
                    for q in gate.target_iter(n_qubits):
                        results[q] = self._collapse(state, q, n_qubits, rng, dtype, device)
                        state = self._last_collapsed
                    continue
                state = self._apply_operation(state, gate, n_qubits, model, dtype, device)
            if not measured:
                for q in report:
                    results[q] = self._collapse(state, q, n_qubits, rng, dtype, device)
                    state = self._last_collapsed
            bits = ['0'] * n_qubits
            for q in report:
                bits[q] = str(results.get(q, 0))
            counts[''.join(reversed(bits))] += 1
        return apply_bit_order(counts, n_qubits, bit_order)

    def _collapse(self, state: torch.Tensor, qubit: int, n_qubits: int, rng: Any,
                  dtype: torch.dtype, device: torch.device) -> int:
        """Measure `qubit` for real: draw an outcome from its marginal, then
        project onto it and renormalize. The collapsed state is left in
        ``_last_collapsed``."""
        dim = 1 << n_qubits
        rho = state.reshape(dim, dim)
        diagonal = torch.diagonal(rho).real
        index = torch.arange(dim, device=diagonal.device)
        p_zero = float(diagonal[(index >> qubit) & 1 == 0].sum())
        p_zero = min(max(p_zero, 0.0), 1.0)
        outcome = 0 if rng.random() < p_zero else 1

        projector = torch.zeros((2, 2), dtype=dtype, device=device)
        projector[outcome, outcome] = 1.0
        collapsed = self._apply(state, torch.kron(projector, projector.conj()),
                                self._wires([qubit], n_qubits), n_qubits)
        trace = torch.diagonal(collapsed.reshape(dim, dim)).sum().real
        self._last_collapsed = collapsed / torch.clamp(trace, min=1e-300)
        return outcome

    # ---------------------------------------------------------------- sampling

    def _sample(self, rho: torch.Tensor, gates: List[Operation], n_qubits: int,
                shots: Optional[int], seed: Optional[int],
                bit_order: str) -> 'typing.Counter[str]':
        from .torch_backend import _collect_measured_qubits, _make_generator

        n_shots = shots if shots is not None else DEFAULT_SHOTS
        probs = torch.diagonal(rho).real.clone()
        # Numerical dust can leave tiny negative diagonal entries; they are not
        # physical and would upset the CDF.
        probs = torch.clamp(probs, min=0.0)
        total = probs.sum()
        if float(total) <= 0.0:
            raise ValueError("The density matrix has zero trace; cannot sample.")
        probs = probs / total

        measured = _collect_measured_qubits(gates, n_qubits)
        keep_mask = (1 << n_qubits) - 1 if measured is None else sum(1 << q for q in measured)

        with torch.no_grad():
            cdf = torch.cumsum(probs, dim=0)
            cdf[-1] = 1.0
            u = torch.rand(n_shots, device=probs.device, dtype=probs.dtype,
                           generator=_make_generator(seed, probs.device))
            samples = torch.searchsorted(cdf, u)
            samples &= keep_mask

        counts: 'typing.Counter[str]' = Counter()
        fmt = f"0{n_qubits}b"
        for idx in samples.tolist():
            counts[format(idx, fmt)] += 1
        return apply_bit_order(counts, n_qubits, bit_order)

    def __repr__(self) -> str:
        return "DensityMatrixBackend()"
