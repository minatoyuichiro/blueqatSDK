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
"""The 'eo' backend: transpile a logical Circuit into exchange pulses.

    import blueqat.eo  # registers the backend
    physical = Circuit(2).h[0].cx[0, 1].run(backend='eo')

Logical qubit i is encoded in physical spins 3i, 3i+1, 3i+2, and the output
is an ordinary Circuit containing only `exch` pulses, runnable on any
simulation backend. All logical gates are exact up to global phase.

Topology note: the emitted pulses assume any pair inside the two triples
involved in a gate can be pulsed (in particular, the Fong-Wandzura CNOT's
bridge pulse connects spin 3c+2 with spin 3t+2, and the encoded SWAP pulses
pair the triples spin-by-spin). This is always fine for simulation; mapping
onto strict nearest-neighbor-only hardware additionally requires dot
orientation assignment and spin-level SWAP routing, which is future work
(cf. exchange-pulse-optimizer).
"""

from typing import Any, List, Optional

from ..backends.backendbase import Backend, register_backend
from ..circuit import Circuit
from ..gate import Operation
from . import sequences


class EOTranspiler(Backend):
    """Transpiler backend converting logical circuits to exchange pulses."""

    _FIXED = {
        'x': sequences.x_sequence,
        'y': sequences.y_sequence,
        'z': sequences.z_sequence,
        'h': sequences.h_sequence,
        's': sequences.s_sequence,
        'sdg': sequences.sdg_sequence,
        't': sequences.t_sequence,
        'tdg': sequences.tdg_sequence,
    }
    _ROTATIONS = {
        'rz': sequences.rz_sequence,
        'phase': sequences.rz_sequence,   # equal to rz up to global phase
        'rx': sequences.rx_sequence,
        'ry': sequences.ry_sequence,
    }

    def run(self, gates: List[Operation], n_qubits: int, *args: Any,
            **kwargs: Any) -> Circuit:
        """`shortest=True` solves each single-qubit gate in closed form rather
        than reading it out of the analytic tables.

        The tables compose known sequences, which is correct and longer than
        necessary: measured, an `rx` costs seven pulses there and three or four
        solved directly, an `ry` nine. A pulse is a gate on this hardware, so
        that is the error budget. It is off by default because it changes the
        emitted pulses for circuits that already work, and a device schedule is
        not something to alter without being asked."""
        from ..gate import GateBlock
        shortest = bool(kwargs.get('shortest', False))
        pulses: List[sequences.Pulse] = []
        # With `shortest`, consecutive one-qubit gates on the same logical
        # qubit are multiplied together and solved once. Solving them one at a
        # time is what makes a run expensive: measured, five gates cost sixteen
        # pulses that way and four as a single matrix. Runs on different
        # triples are not interleaved here, which is sound because exchange
        # pulses on disjoint triples commute.
        pending: dict = {}

        def flush(qubit: Optional[int] = None) -> None:
            for target in (sorted(pending) if qubit is None
                           else ([qubit] if qubit in pending else [])):
                pulses.extend(self._solve_matrix(pending.pop(target), 3 * target))

        for gate in gates:
            name = gate.lowername
            if isinstance(gate, GateBlock):
                # ブロックは中身を展開してトランスパイルする
                sub = self.run(gate.ops, n_qubits)
                for op in sub.ops:
                    i, j = op.targets
                    pulses.append(((int(i), int(j)), float(op.theta)))
                continue
            if name in ('i', 'barrier'):
                continue
            if name in self._FIXED or name in self._ROTATIONS:
                for t in gate.target_iter(n_qubits):
                    if shortest:
                        matrix = _matrix_of(gate)
                        pending[t] = (matrix if t not in pending
                                      else matrix @ pending[t])
                    elif name in self._FIXED:
                        pulses += self._FIXED[name](offset=3 * t)
                    else:
                        pulses += self._ROTATIONS[name](gate.theta, offset=3 * t)
            elif name == 'cx':
                for c, t in gate.control_target_iter(n_qubits):
                    flush(c); flush(t)
                    pulses += sequences.cx_sequence(3 * c, 3 * t)
            elif name == 'cz':
                for c, t in gate.control_target_iter(n_qubits):
                    flush(c); flush(t)
                    pulses += sequences.cz_sequence(3 * c, 3 * t)
            elif name == 'swap':
                for a, b in gate.control_target_iter(n_qubits):
                    flush(a); flush(b)
                    pulses += sequences.swap_sequence(3 * a, 3 * b)
            else:
                raise ValueError(
                    f"Gate '{name}' is not supported by the exchange-only "
                    "transpiler. Decompose it into "
                    "x/y/z/h/s/t/rx/ry/rz/cx/cz/swap first.")
        flush()
        return sequences.sequence_to_circuit(pulses, 3 * n_qubits)

    @staticmethod
    def _solve_matrix(matrix, offset: int) -> List[sequences.Pulse]:
        """One accumulated one-qubit matrix, solved directly.

        An accumulated identity needs no pulses at all. That is worth stating
        rather than leaving to arithmetic: a run that cancels out costs nothing
        here, and cost a full table entry per gate before.
        """
        import torch
        from .optimizer import decompose_1q
        eye = torch.eye(2, dtype=torch.complex128)
        head = complex(matrix[0, 0])
        if abs(head) < 1e-12:
            head = complex(matrix[0, 1])
        if abs(head) > 1e-12:
            aligned = matrix * (abs(head) / head)
            if torch.allclose(aligned, eye, atol=1e-12):
                return []
        return decompose_1q(matrix, offset=offset)


def _matrix_of(gate: Operation):
    """The 2x2 matrix of a one-qubit gate, read by running it on one qubit."""
    import copy
    import torch
    from ..circuit import Circuit
    from ..circuit_funcs import circuit_to_unitary
    moved = copy.copy(gate)
    moved.targets = (0, )
    single = Circuit(1)
    single.ops.append(moved)
    return torch.as_tensor(circuit_to_unitary(single), dtype=torch.complex128)


register_backend('eo', EOTranspiler, overwrite=True)
