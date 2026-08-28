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
    ``noise=`` to ``Circuit.run``, which routes here automatically.
    """

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
        noise_scale = kwargs.get("noise_scale", 1.0)
        if model is not None and float(noise_scale) != 1.0:
            model = model.scaled(float(noise_scale))
        elif model is None and float(noise_scale) != 1.0:
            raise ValueError("noise_scale= was given without noise=; there is nothing "
                             "to scale.")

        state = self._initial_state(kwargs.get("initial"), n_qubits, dtype, device)
        state = self._run_gates(state, gates, n_qubits, model, dtype, device)

        rho = state.reshape(1 << n_qubits, 1 << n_qubits)

        if hamiltonian is not None:
            from ..utils import pauli_expectation
            return pauli_expectation(hamiltonian, rho, n_qubits)

        if shots is None and returns != "shots":
            return rho

        return self._sample(rho, gates, n_qubits, shots, seed, bit_order)

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
