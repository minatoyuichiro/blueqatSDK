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
"""Stabilizer simulation: Clifford circuits at thousands of qubits.

A stabilizer state is stored by its stabilizer generators rather than its
amplitudes, so memory is ``O(n**2)`` bits instead of ``2**n`` complex numbers.
That is the difference between 12 qubits and 12000 -- and it is what makes
error-correction work possible at all, since even a distance-5 surface code
needs 49 qubits before any noise is added.

The price is that only Clifford gates, measurement and reset are allowed
(Gottesman-Knill). For anything with a ``t`` or an ``rx`` in it, use the
statevector or density-matrix backends.

    Circuit(200).h[0].cx[0, 1] ... .m[:].run(backend='stabilizer', shots=100)

The state is a tableau of `2n` rows -- `n` destabilizer generators and `n`
stabilizer generators -- which is precisely the representation
:class:`blueqat.clifford.Clifford` already uses for an operator, since a
state's generators transform under a gate exactly as an operator's rows do.
Measurement is the part that is new here, and follows Aaronson and Gottesman
(quant-ph/0406196).
"""

import random
from collections import Counter
import typing
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .circuit import Circuit
from .clifford import Clifford, _CLIFFORD_REWRITES, _PRIMITIVE, _pauli_mul
from .backends.backendbase import Backend, BIT_ORDERS, apply_bit_order

__all__ = ['StabilizerSimulator', 'StabilizerBackend']

DEFAULT_SHOTS: int = 1024


def _popcount(mask: int) -> int:
    return bin(mask).count('1')


class StabilizerSimulator:
    """A stabilizer state under Clifford gates, measurement and reset.

    Starts in ``|0...0>``. Gates are applied by name (see
    :data:`blueqat.clifford.SHIFT_RULE_GATES` for the Clifford set), and
    `measure` collapses a qubit, returning the outcome.
    """

    def __init__(self, n_qubits: int, seed: Optional[int] = None) -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be at least 1, got {n_qubits}.")
        self.n_qubits = n_qubits
        # Rows 0..n-1 destabilize, rows n..2n-1 stabilize; for |0...0> those are
        # X_i and Z_i, which is exactly the identity tableau.
        self.tableau = Clifford.identity(n_qubits)
        self._scratch: Tuple[int, int, int] = (0, 0, 0)
        self._rng = random.Random(seed)
        self._mask = (1 << n_qubits) - 1

    def copy(self) -> 'StabilizerSimulator':
        clone = StabilizerSimulator.__new__(StabilizerSimulator)
        clone.n_qubits = self.n_qubits
        clone.tableau = self.tableau.copy()
        clone._scratch = self._scratch
        clone._rng = random.Random()
        clone._rng.setstate(self._rng.getstate())
        clone._mask = self._mask
        return clone

    # -------------------------------------------------------------- gates

    def apply(self, name: str, qubits: Sequence[int]) -> None:
        """Apply a Clifford gate by name to `qubits`."""
        for primitive, targets in _expand(name, tuple(qubits)):
            self.tableau.apply_primitive(primitive, targets)

    # -------------------------------------------------------- measurement

    def _row(self, index: int) -> Tuple[int, int, int]:
        if index == 2 * self.n_qubits:
            return self._scratch
        t = self.tableau
        return (t.x[index], t.z[index], t.phase[index])

    def _set_row(self, index: int, row: Tuple[int, int, int]) -> None:
        if index == 2 * self.n_qubits:
            self._scratch = row
            return
        t = self.tableau
        t.x[index], t.z[index], t.phase[index] = row

    def _rowsum(self, target: int, source: int) -> None:
        """``row[target] <- row[source] * row[target]``, tracking the sign.

        Aaronson and Gottesman give this as a four-case sum over the qubits, but
        their tableau writes a doubly-set column as ``X Z`` where this one writes
        ``Y``. Rather than port the sum into the other convention, each row is
        converted to a power of i in front of ``prod X**x Z**z``, multiplied with
        the shared Pauli routine, and converted back -- the two conventions differ
        by exactly ``i**popcount(x & z)``.
        """
        xs, zs, rs = self._row(source)
        xt, zt, rt = self._row(target)
        a_source = (2 * rs + _popcount(xs & zs)) % 4
        a_target = (2 * rt + _popcount(xt & zt)) % 4

        x, z, ipow = _pauli_mul(xs, zs, a_source, xt, zt, a_target)
        ipow = (ipow - _popcount(x & z)) % 4
        if ipow % 2:
            raise AssertionError(
                f"rowsum produced an odd power of i ({ipow}), which a product of "
                f"Hermitian Pauli words cannot have.")
        self._set_row(target, (x, z, ipow // 2))

    def measure(self, qubit: int) -> int:
        """Measure `qubit` in the Z basis, collapsing the state. Returns 0 or 1."""
        if not 0 <= qubit < self.n_qubits:
            raise ValueError(f"qubit must be in range(0, {self.n_qubits}), got {qubit}.")
        n = self.n_qubits
        bit = 1 << qubit

        pivot = None
        for p in range(n, 2 * n):
            if self.tableau.x[p] & bit:
                pivot = p
                break

        if pivot is None:
            # No stabilizer anticommutes with Z on this qubit, so the outcome is
            # already determined; multiply the relevant generators together in the
            # scratch row and read off its sign.
            self._scratch = (0, 0, 0)
            for i in range(n):
                if self.tableau.x[i] & bit:
                    self._rowsum(2 * n, i + n)
            return self._scratch[2]

        for i in range(2 * n):
            # `pivot - n` is the one row that anticommutes with `pivot`, so their
            # product is anti-Hermitian and has no place in a tableau of signed
            # Pauli words. It is also the one row whose value is about to be
            # overwritten, so the multiplication would be discarded anyway --
            # skipping it is exactly equivalent, and keeps every rowsum a product
            # of commuting rows.
            if i != pivot and i != pivot - n and (self.tableau.x[i] & bit):
                self._rowsum(i, pivot)
        # The old stabilizer becomes the destabilizer of the new one.
        self._set_row(pivot - n, self._row(pivot))
        outcome = self._rng.getrandbits(1)
        self._set_row(pivot, (0, bit, outcome))
        return outcome

    def reset(self, qubit: int) -> None:
        """Force `qubit` back to ``|0>``."""
        if self.measure(qubit):
            self.apply('x', (qubit, ))

    # ------------------------------------------------------------ readout

    def stabilizers(self) -> List[str]:
        """The state's stabilizer generators, as signed Pauli strings.

        Character `q` of each string is the Pauli on qubit `q`.
        """
        n = self.n_qubits
        out = []
        for r in range(n, 2 * n):
            x, z, phase = self.tableau.x[r], self.tableau.z[r], self.tableau.phase[r]
            letters = []
            for q in range(n):
                xq, zq = (x >> q) & 1, (z >> q) & 1
                letters.append('Y' if xq and zq else 'IXZ'[xq + 2 * zq])
            out.append(('-' if phase else '+') + ''.join(letters))
        return out

    def __repr__(self) -> str:
        return f"StabilizerSimulator({self.n_qubits}, {self.stabilizers()})"


def _expand(name: str, qubits: Tuple[int, ...]) -> List[Tuple[str, Tuple[int, ...]]]:
    """A gate name and its qubits as tableau primitives."""
    name = name.lower()
    if name in _PRIMITIVE:
        return [(name, qubits)]
    if name in _CLIFFORD_REWRITES:
        return list(_CLIFFORD_REWRITES[name](qubits))
    raise ValueError(
        f"{name} is not a Clifford gate; the stabilizer backend cannot run it. "
        f"Use the statevector or density-matrix backend instead.")


class StabilizerBackend(Backend):
    """The ``'stabilizer'`` backend: Clifford circuits, at scale.

    ``shots=N`` returns a Counter of bitstrings, with the same ``seed`` and
    ``bit_order`` arguments as the other backends. Without ``shots``, the
    :class:`StabilizerSimulator` itself is returned, so its stabilizer
    generators can be inspected.
    """

    def run(self, gates: List[Any], n_qubits: int, *args: Any, **kwargs: Any) -> Any:
        from .gate import GateBlock

        n_qubits = max(n_qubits, 1)
        shots = kwargs.get("shots")
        returns = kwargs.get("returns")
        seed = kwargs.get("seed")
        bit_order = kwargs.get("bit_order", "q0_last")
        if bit_order not in BIT_ORDERS:
            raise ValueError(f"bit_order must be one of {BIT_ORDERS}, got {bit_order!r}.")
        if returns in ("statevector", "amplitude", "statevector_and_shots"):
            raise ValueError(
                f"returns={returns!r} asks for amplitudes, which a stabilizer state "
                f"does not carry. Use the statevector backend for that.")

        from .circuit_funcs.flatten import flatten
        ops = list(flatten(Circuit(n_qubits, list(gates))).ops)
        measured = sorted({q for op in ops if op.lowername == 'measure'
                           for q in _targets(op)})

        if shots is None and returns != "shots":
            state = StabilizerSimulator(n_qubits, seed=seed)
            self._run_once(state, ops, n_qubits)
            return state

        n_shots = shots if shots is not None else DEFAULT_SHOTS
        rng = random.Random(seed)
        report = measured if measured else list(range(n_qubits))
        counts: 'typing.Counter[str]' = Counter()
        for _ in range(n_shots):
            # Each shot is an independent trajectory: measurement outcomes are
            # genuinely random, so there is no final state to sample from.
            state = StabilizerSimulator(n_qubits, seed=rng.getrandbits(63))
            results = self._run_once(state, ops, n_qubits)
            if not measured:
                results = {q: state.measure(q) for q in report}
            bits = ['0'] * n_qubits
            for q in report:
                bits[q] = str(results.get(q, 0))
            counts[''.join(reversed(bits))] += 1
        return apply_bit_order(counts, n_qubits, bit_order)

    @staticmethod
    def _run_once(state: StabilizerSimulator, ops: List[Any],
                  n_qubits: int) -> Dict[int, int]:
        results: Dict[int, int] = {}
        for op in ops:
            name = op.lowername
            if name == 'barrier':
                continue
            if name == 'measure':
                for q in _targets(op):
                    results[q] = state.measure(q)
            elif name == 'reset':
                for q in _targets(op):
                    state.reset(q)
            else:
                state.apply(name, _targets(op))
        return results

    def __repr__(self) -> str:
        return "StabilizerBackend()"


def _targets(op: Any) -> Tuple[int, ...]:
    targets = op.targets
    if isinstance(targets, int):
        return (targets, )
    return tuple(int(t) for t in targets)
