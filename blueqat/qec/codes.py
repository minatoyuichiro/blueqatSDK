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
"""Stabilizer codes: generators, logical operators and qubit layout."""

from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ['StabilizerCode', 'repetition_code', 'rotated_surface_code']

_PAULIS = frozenset('IXYZ')


def _commute(a: str, b: str) -> bool:
    """Whether two Pauli strings commute (they anticommute on an odd number of
    qubits, or they do not)."""
    disagreements = sum(1 for p, q in zip(a, b) if p != 'I' and q != 'I' and p != q)
    return disagreements % 2 == 0


def _weight(pauli: str) -> int:
    return sum(1 for p in pauli if p != 'I')


class StabilizerCode:
    """A stabilizer code: its generators, its logical operators and its layout.

    Pauli strings are indexed by qubit -- character `q` acts on data qubit `q` --
    which is the same convention as ``bit_order='q0_first'`` and avoids the
    reversal mistakes that bite when reading a syndrome.

    `ancilla_of` maps each stabilizer index to the qubit that measures it in a
    syndrome-extraction circuit; ancillas are numbered from `n_data` upward in
    stabilizer order, so a circuit built from this code has
    ``n_data + len(stabilizers)`` qubits in total.
    """

    def __init__(self, name: str, n_data: int, stabilizers: Sequence[str],
                 logical_x: Sequence[str], logical_z: Sequence[str],
                 distance: Optional[int] = None,
                 coordinates: Optional[Dict[int, Tuple[int, int]]] = None) -> None:
        for label, group in (('stabilizer', stabilizers), ('logical_x', logical_x),
                             ('logical_z', logical_z)):
            for pauli in group:
                if len(pauli) != n_data:
                    raise ValueError(f"{label} {pauli!r} has length {len(pauli)}, "
                                     f"expected {n_data}.")
                if set(pauli) - _PAULIS:
                    raise ValueError(f"{label} {pauli!r} has characters outside IXYZ.")
        self.name = name
        self.n_data = n_data
        self.stabilizers = list(stabilizers)
        self.logical_x = list(logical_x)
        self.logical_z = list(logical_z)
        self.distance = distance
        self.coordinates = dict(coordinates) if coordinates else {}

    # ------------------------------------------------------------- layout

    @property
    def n_stabilizers(self) -> int:
        return len(self.stabilizers)

    @property
    def n_logical(self) -> int:
        return len(self.logical_x)

    @property
    def data_qubits(self) -> List[int]:
        return list(range(self.n_data))

    @property
    def ancilla_qubits(self) -> List[int]:
        return [self.n_data + i for i in range(self.n_stabilizers)]

    @property
    def n_qubits(self) -> int:
        """Data plus one ancilla per stabilizer."""
        return self.n_data + self.n_stabilizers

    def ancilla_of(self, stabilizer_index: int) -> int:
        if not 0 <= stabilizer_index < self.n_stabilizers:
            raise ValueError(f"stabilizer index must be in "
                             f"range(0, {self.n_stabilizers}).")
        return self.n_data + stabilizer_index

    def support(self, stabilizer_index: int) -> List[Tuple[int, str]]:
        """``(qubit, pauli)`` pairs a stabilizer acts on, in qubit order."""
        pauli = self.stabilizers[stabilizer_index]
        return [(q, p) for q, p in enumerate(pauli) if p != 'I']

    # --------------------------------------------------------- validation

    def check(self) -> None:
        """Raise unless this really is a stabilizer code.

        Generators must commute with each other and with every logical, and
        logical X_i must anticommute with logical Z_i and commute with the rest.
        A layout mistake shows up here rather than as a silently wrong threshold.
        """
        for i, a in enumerate(self.stabilizers):
            for b in self.stabilizers[i + 1:]:
                if not _commute(a, b):
                    raise ValueError(f"stabilizers {a!r} and {b!r} do not commute.")
        for logical in self.logical_x + self.logical_z:
            for stabilizer in self.stabilizers:
                if not _commute(logical, stabilizer):
                    raise ValueError(f"logical {logical!r} does not commute with "
                                     f"stabilizer {stabilizer!r}.")
        for i, lx in enumerate(self.logical_x):
            for j, lz in enumerate(self.logical_z):
                if _commute(lx, lz) != (i != j):
                    raise ValueError(
                        f"logical_x[{i}] and logical_z[{j}] have the wrong "
                        f"commutation: they must anticommute exactly when i == j.")

    def logical_weight(self) -> int:
        """The lowest weight of any logical operator, i.e. the code distance.

        Brute force over all Pauli strings, so only usable for small codes -- but
        that is exactly where a hand-written layout needs checking.
        """
        if self.n_data > 12:
            raise ValueError(f"logical_weight enumerates 4**{self.n_data} operators; "
                             f"pass the distance explicitly for a code this size.")
        best = None
        for code in range(4 ** self.n_data):
            pauli = []
            value = code
            for _ in range(self.n_data):
                pauli.append('IXYZ'[value & 3])
                value >>= 2
            candidate = ''.join(pauli)
            if not all(_commute(candidate, s) for s in self.stabilizers):
                continue
            # Commutes with every stabilizer: it is either in the stabilizer
            # group (trivial) or a logical operator.
            if all(_commute(candidate, l) for l in self.logical_x + self.logical_z):
                continue
            weight = _weight(candidate)
            if best is None or weight < best:
                best = weight
        return best if best is not None else 0

    def __repr__(self) -> str:
        return (f"StabilizerCode({self.name!r}, n_data={self.n_data}, "
                f"stabilizers={self.n_stabilizers}, distance={self.distance})")


def _pauli_string(n: int, assignment: Dict[int, str]) -> str:
    return ''.join(assignment.get(q, 'I') for q in range(n))


def repetition_code(distance: int) -> StabilizerCode:
    """The `d`-qubit bit-flip repetition code.

    Stabilizers are ``Z_i Z_{i+1}``, so a single X error lights the one or two
    checks either side of it. Logical Z is a single ``Z``; logical X is ``X`` on
    every qubit. It protects against X errors only -- which makes it the
    standard first target for a circuit-level study, because everything about
    the syndrome circuit and the decoder is the same as for a real code while
    the answer stays checkable by hand.
    """
    if distance < 2:
        raise ValueError(f"distance must be at least 2, got {distance}.")
    n = distance
    stabilizers = [_pauli_string(n, {i: 'Z', i + 1: 'Z'}) for i in range(n - 1)]
    logical_x = ['X' * n]
    logical_z = [_pauli_string(n, {0: 'Z'})]
    coordinates = {q: (q, 0) for q in range(n)}
    code = StabilizerCode(f'repetition-{distance}', n, stabilizers, logical_x,
                          logical_z, distance=distance, coordinates=coordinates)
    code.check()
    return code


def rotated_surface_code(distance: int) -> StabilizerCode:
    """The rotated surface code of odd distance `d`, on ``d*d`` data qubits.

    Data qubit ``(x, y)`` is index ``y * d + x``. The bulk carries weight-4
    plaquettes on every ``2x2`` square, alternating type with the parity of
    ``x + y``; the boundaries carry weight-2 checks, Z along the top and bottom
    rows and X along the left and right columns. Logical Z runs down a column
    and logical X across a row, so each has weight `d`.
    """
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be an odd number at least 3, got {distance}.")
    d = distance
    index = lambda x, y: y * d + x
    n = d * d
    z_checks: List[str] = []
    x_checks: List[str] = []

    for y in range(d - 1):
        for x in range(d - 1):
            corners = {index(x, y): '', index(x + 1, y): '',
                       index(x, y + 1): '', index(x + 1, y + 1): ''}
            kind = 'Z' if (x + y) % 2 == 0 else 'X'
            pauli = _pauli_string(n, {q: kind for q in corners})
            (z_checks if kind == 'Z' else x_checks).append(pauli)

    # Weight-2 boundary checks. Their parity is chosen so that each one sits
    # against a bulk plaquette of the other type, which is what keeps every
    # generator commuting.
    for x in range(d - 1):
        if x % 2 == 1:
            z_checks.append(_pauli_string(n, {index(x, 0): 'Z', index(x + 1, 0): 'Z'}))
        else:
            z_checks.append(_pauli_string(
                n, {index(x, d - 1): 'Z', index(x + 1, d - 1): 'Z'}))
    for y in range(d - 1):
        if y % 2 == 0:
            x_checks.append(_pauli_string(n, {index(0, y): 'X', index(0, y + 1): 'X'}))
        else:
            x_checks.append(_pauli_string(
                n, {index(d - 1, y): 'X', index(d - 1, y + 1): 'X'}))

    logical_z = [_pauli_string(n, {index(0, y): 'Z' for y in range(d)})]
    logical_x = [_pauli_string(n, {index(x, 0): 'X' for x in range(d)})]
    coordinates = {index(x, y): (x, y) for y in range(d) for x in range(d)}
    code = StabilizerCode(f'rotated-surface-{d}', n, z_checks + x_checks,
                          logical_x, logical_z, distance=d, coordinates=coordinates)
    code.check()
    return code
