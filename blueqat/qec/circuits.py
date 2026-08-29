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
"""Syndrome-extraction circuits for stabilizer codes."""

from typing import Callable, List, Optional, Sequence, Tuple

from ..circuit import Circuit
from .codes import StabilizerCode

__all__ = ['syndrome_round', 'syndrome_extraction_circuit', 'index_order']


def index_order(code: StabilizerCode, stabilizer_index: int) -> List[int]:
    """The default interaction order: ascending data-qubit index.

    Deterministic, and deliberately plain. On the surface code the order in
    which an ancilla touches its four data qubits decides which two-qubit
    errors propagate into weight-2 data errors -- "hook" errors -- and so
    decides whether the circuit-level distance is `d` or only ``(d+1)/2``. A
    schedule chosen for that reason is a property of the experiment, not of the
    code, so it is passed in rather than assumed here.
    """
    return [q for q, _ in code.support(stabilizer_index)]


def syndrome_round(code: StabilizerCode, round_index: int = 0,
                   order: Optional[Callable[[StabilizerCode, int], Sequence[int]]] = None,
                   reset_ancillas: bool = True) -> Circuit:
    """One round of syndrome extraction, as a circuit.

    Each stabilizer is measured by its own ancilla, prepared in ``|+>``, coupled
    to the data qubits by a controlled Pauli, rotated back and measured. The
    measurement is keyed ``"s{index}_r{round}"`` so rounds can be told apart in
    the results.

    `order` picks the interaction order per stabilizer (see :func:`index_order`).
    `reset_ancillas` returns the ancillas to ``|0>`` afterwards, which is what
    lets the next round reuse them.
    """
    order = order or index_order
    circuit = Circuit(code.n_qubits)
    for index in range(code.n_stabilizers):
        ancilla = code.ancilla_of(index)
        support = dict(code.support(index))
        circuit.h[ancilla]
        for qubit in order(code, index):
            pauli = support.get(qubit)
            if pauli is None:
                raise ValueError(
                    f"order() returned qubit {qubit}, which stabilizer {index} "
                    f"does not act on.")
            # Controlled-P from the ancilla: the standard way to read out an
            # eigenvalue without disturbing the codespace.
            if pauli == 'X':
                circuit.cx[ancilla, qubit]
            elif pauli == 'Z':
                circuit.cz[ancilla, qubit]
            else:  # 'Y'
                circuit.cy[ancilla, qubit]
        circuit.h[ancilla]
        circuit.m(key=f"s{index}_r{round_index}")[ancilla]
        if reset_ancillas:
            circuit.reset[ancilla]
    return circuit


def syndrome_extraction_circuit(
        code: StabilizerCode, rounds: int = 1,
        order: Optional[Callable[[StabilizerCode, int], Sequence[int]]] = None) -> Circuit:
    """`rounds` back-to-back rounds of syndrome extraction."""
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}.")
    circuit = Circuit(code.n_qubits)
    for r in range(rounds):
        circuit += syndrome_round(code, round_index=r, order=order)
    return circuit
