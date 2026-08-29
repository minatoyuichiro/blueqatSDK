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
"""Clifford group elements: uniform sampling, composition, inverse, synthesis.

A Clifford operator is stored as a stabilizer tableau -- the images of the
generators ``X_0..X_{n-1}`` and ``Z_0..Z_{n-1}`` under conjugation -- which makes
composition and inversion cheap and exact, with no ``2**n`` matrices anywhere.

This is what randomized benchmarking needs: draw uniform random Cliffords,
compose the sequence, and invert it to get the single recovery operator::

    from blueqat.clifford import random_clifford, Clifford

    seq = [random_clifford(2, seed=i) for i in range(m)]
    total = Clifford.identity(2)
    for c in seq:
        total = total.then(c)
    circuit = sum((c.to_circuit() for c in seq), Circuit(2))
    circuit += total.inverse().to_circuit()   # ...back to |00>

Recovering with ``total.inverse()`` gives one Clifford, not a replay of the
sequence backwards, which is what keeps the benchmarking decay curve meaningful.
"""

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from blueqat import Circuit

__all__ = ['Clifford', 'random_clifford']

# Gates whose action on a Pauli is expressed directly on the tableau. Everything
# else Clifford is rewritten into these by `_CLIFFORD_REWRITES`.
_PRIMITIVE = ('h', 's', 'cx', 'x', 'y', 'z')

# name -> callable(targets) -> list of (primitive, qubits)
_CLIFFORD_REWRITES: Dict[str, Any] = {
    'i': lambda q: [],
    'barrier': lambda q: [],
    'sdg': lambda q: [('s', q)] * 3,
    'sx': lambda q: [('h', q), ('s', q), ('h', q)],
    'sxdg': lambda q: [('h', q), ('s', q), ('s', q), ('s', q), ('h', q)],
    'cnot': lambda q: [('cx', q)],
    'cz': lambda q: [('h', (q[1],)), ('cx', q), ('h', (q[1],))],
    'cy': lambda q: [('s', (q[1],))] * 3 + [('cx', q), ('s', (q[1],))],
    'swap': lambda q: [('cx', q), ('cx', (q[1], q[0])), ('cx', q)],
}


def _parity(mask: int) -> int:
    return bin(mask).count('1') & 1


class Clifford:
    """An `n`-qubit Clifford operator, as the images of ``X_i`` and ``Z_i``.

    Row ``i`` holds the image of ``X_i`` and row ``n + i`` the image of ``Z_i``,
    each as bit masks `x` and `z` plus a sign bit: bit `q` set in both means a
    ``Y`` on qubit `q`. Global phase is not tracked -- it is not observable and
    not part of the Clifford group as benchmarking uses it.
    """

    __slots__ = ('n_qubits', 'x', 'z', 'phase')

    def __init__(self, n_qubits: int, x: Sequence[int], z: Sequence[int],
                 phase: Sequence[int]) -> None:
        self.n_qubits = n_qubits
        self.x = list(x)
        self.z = list(z)
        self.phase = list(phase)

    # ------------------------------------------------------------ construction

    @classmethod
    def identity(cls, n_qubits: int) -> 'Clifford':
        x = [1 << i for i in range(n_qubits)] + [0] * n_qubits
        z = [0] * n_qubits + [1 << i for i in range(n_qubits)]
        return cls(n_qubits, x, z, [0] * (2 * n_qubits))

    @classmethod
    def from_circuit(cls, circuit: Circuit, n_qubits: Optional[int] = None) -> 'Clifford':
        """The Clifford a circuit implements. Raises on a non-Clifford gate."""
        n = circuit.n_qubits if n_qubits is None else n_qubits
        n = max(n, 1)
        tableau = cls.identity(n)
        for primitive, qubits in _primitive_ops(circuit, n):
            tableau.apply_primitive(primitive, qubits)
        return tableau

    def copy(self) -> 'Clifford':
        return Clifford(self.n_qubits, self.x, self.z, self.phase)

    # ------------------------------------------------------- tableau updates

    def apply_primitive(self, name: str, qubits: Sequence[int]) -> None:
        """Conjugate every stored row by `name`, i.e. left-multiply this operator
        by that gate.

        The same update also advances a stabilizer *state*, since a state's
        stabilizer generators transform by conjugation exactly as an operator's
        rows do -- which is what :mod:`blueqat.stabilizer` builds on.
        """
        if name == 'h':
            (q,) = qubits
            bit = 1 << q
            for r in range(2 * self.n_qubits):
                xq, zq = (self.x[r] >> q) & 1, (self.z[r] >> q) & 1
                self.phase[r] ^= xq & zq
                if xq != zq:
                    self.x[r] ^= bit
                    self.z[r] ^= bit
        elif name == 's':
            (q,) = qubits
            bit = 1 << q
            for r in range(2 * self.n_qubits):
                xq, zq = (self.x[r] >> q) & 1, (self.z[r] >> q) & 1
                self.phase[r] ^= xq & zq
                if xq:
                    self.z[r] ^= bit
        elif name == 'cx':
            a, b = qubits
            for r in range(2 * self.n_qubits):
                xa, za = (self.x[r] >> a) & 1, (self.z[r] >> a) & 1
                xb, zb = (self.x[r] >> b) & 1, (self.z[r] >> b) & 1
                self.phase[r] ^= xa & zb & (xb ^ za ^ 1)
                if xa:
                    self.x[r] ^= 1 << b
                if zb:
                    self.z[r] ^= 1 << a
        elif name == 'x':
            (q,) = qubits
            for r in range(2 * self.n_qubits):
                self.phase[r] ^= (self.z[r] >> q) & 1
        elif name == 'z':
            (q,) = qubits
            for r in range(2 * self.n_qubits):
                self.phase[r] ^= (self.x[r] >> q) & 1
        elif name == 'y':
            (q,) = qubits
            for r in range(2 * self.n_qubits):
                self.phase[r] ^= ((self.x[r] >> q) & 1) ^ ((self.z[r] >> q) & 1)
        else:
            raise ValueError(f"{name} is not a tableau primitive.")

    # -------------------------------------------------------------- algebra

    def then(self, other: 'Clifford') -> 'Clifford':
        """The Clifford that applies `self` first and then `other`."""
        if self.n_qubits != other.n_qubits:
            raise ValueError(f"Cannot compose a {self.n_qubits}-qubit Clifford with a "
                             f"{other.n_qubits}-qubit one.")
        n = self.n_qubits
        out_x, out_z, out_phase = [], [], []
        for r in range(2 * n):
            x, z, ipow = 0, 0, 2 * self.phase[r]
            for q in range(n):
                xq, zq = (self.x[r] >> q) & 1, (self.z[r] >> q) & 1
                if xq and zq:
                    # Y_q = i X_q Z_q, so its image carries that same factor.
                    ipow += 1
                if xq:
                    x, z, ipow = _pauli_mul(x, z, ipow, other.x[q], other.z[q],
                                            _row_ipow(other, q))
                if zq:
                    x, z, ipow = _pauli_mul(x, z, ipow, other.x[n + q], other.z[n + q],
                                            _row_ipow(other, n + q))
            # Back to the tableau's own convention, where a doubly-set bit is Y.
            ipow = (ipow - _popcount(x & z)) % 4
            assert ipow % 2 == 0, "a Hermitian Pauli cannot pick up an odd power of i"
            out_x.append(x)
            out_z.append(z)
            out_phase.append((ipow // 2) % 2)
        return Clifford(n, out_x, out_z, out_phase)

    def inverse(self) -> 'Clifford':
        """The inverse Clifford."""
        return Clifford.from_circuit(self._circuit_from(self._reduction_gates()),
                                     self.n_qubits)

    # ------------------------------------------------------------- synthesis

    def to_circuit(self) -> Circuit:
        """A circuit of ``h``, ``s``, ``sdg``, ``cx``, ``x`` and ``z`` implementing this
        Clifford (up to global phase)."""
        gates = self._reduction_gates()
        # The reduction found G_m...G_1 U = I, so U = G_1^-1 ... G_m^-1: run the
        # recorded gates backwards, inverting each. (The forward list, unreversed,
        # is exactly the inverse operator -- which is what `inverse` uses.)
        inverted = [(_INVERSE[name], qubits) for name, qubits in reversed(gates)]
        return self._circuit_from(inverted)

    def _circuit_from(self, gates: Sequence[Tuple[str, Sequence[int]]]) -> Circuit:
        circuit = Circuit(self.n_qubits)
        for name, qubits in gates:
            if name == 'cx':
                circuit.cx[qubits[0], qubits[1]]
            else:
                getattr(circuit, name)[qubits[0]]
        return circuit

    def _reduction_gates(self) -> List[Tuple[str, Tuple[int, ...]]]:
        """Gates reducing this tableau to the identity, in the order applied.

        Sweeping qubit by qubit: first make the image of ``X_i`` be exactly
        ``X_i``, then make the image of ``Z_i`` be exactly ``Z_i`` using only
        operations that leave the first alone. Once qubit `i` is fixed, every
        later row commutes with ``X_i`` and ``Z_i`` and so has no support there,
        which is why the sweep never has to revisit it.
        """
        work = self.copy()
        n = work.n_qubits
        gates: List[Tuple[str, Tuple[int, ...]]] = []

        def emit(name: str, *qubits: int) -> None:
            gates.append((name, tuple(qubits)))
            work.apply_primitive(name, qubits)

        for i in range(n):
            dest, stab = i, n + i

            # --- image of X_i  ->  X_i -------------------------------------
            if not (work.x[dest] >> i) & 1:
                # Bring an X (or, via H, a Z) onto qubit i.
                for j in range(i, n):
                    if (work.x[dest] >> j) & 1:
                        if j != i:
                            emit('cx', j, i)
                        break
                else:
                    for j in range(i, n):
                        if (work.z[dest] >> j) & 1:
                            emit('h', j)
                            if j != i:
                                emit('cx', j, i)
                            break
            for j in range(i + 1, n):
                if (work.x[dest] >> j) & 1:
                    emit('cx', i, j)
            if work.z[dest]:
                if not (work.z[dest] >> i) & 1:
                    emit('s', i)
                for j in range(i + 1, n):
                    if (work.z[dest] >> j) & 1:
                        emit('cx', j, i)
                emit('s', i)

            # --- image of Z_i  ->  Z_i -------------------------------------
            # Only gates avoiding qubit i, plus cx(j, i), leave the image of X_i
            # untouched; the one exception is h-s-h on qubit i, which fixes X and
            # sends Y to Z -- exactly what a leftover Y on qubit i needs.
            for j in range(i + 1, n):
                if (work.x[stab] >> j) & 1:
                    if (work.z[stab] >> j) & 1:
                        emit('s', j)
                    emit('h', j)
            if (work.x[stab] >> i) & 1:
                emit('h', i)
                emit('s', i)
                emit('h', i)
            for j in range(i + 1, n):
                if (work.z[stab] >> j) & 1:
                    emit('cx', j, i)

        # --- signs ---------------------------------------------------------
        for i in range(n):
            if work.phase[i]:
                emit('z', i)
            if work.phase[n + i]:
                emit('x', i)

        assert work == Clifford.identity(n), "Clifford reduction did not reach the identity"
        return gates

    # ------------------------------------------------------------- protocol

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Clifford):
            return NotImplemented
        return (self.n_qubits == other.n_qubits and self.x == other.x
                and self.z == other.z and self.phase == other.phase)

    def __hash__(self) -> int:
        return hash((self.n_qubits, tuple(self.x), tuple(self.z), tuple(self.phase)))

    def __repr__(self) -> str:
        rows = []
        for r in range(2 * self.n_qubits):
            label = f"X{r}" if r < self.n_qubits else f"Z{r - self.n_qubits}"
            rows.append(f"{label}->{'-' if self.phase[r] else '+'}"
                        f"{_pauli_string(self.x[r], self.z[r], self.n_qubits)}")
        return f"Clifford({self.n_qubits}, {' '.join(rows)})"


_INVERSE = {'h': 'h', 's': 'sdg', 'sdg': 's', 'cx': 'cx', 'x': 'x', 'z': 'z', 'y': 'y'}


def _popcount(mask: int) -> int:
    return bin(mask).count('1')


def _pauli_string(x: int, z: int, n: int) -> str:
    out = []
    for q in range(n):
        xq, zq = (x >> q) & 1, (z >> q) & 1
        out.append('IXZY'[xq + 2 * zq] if not (xq and zq) else 'Y')
    return ''.join(out)


def _row_ipow(tableau: 'Clifford', row: int) -> int:
    """A tableau row as a power of i in front of ``prod X**x Z**z``.

    The row's own convention writes a doubly-set bit as ``Y``, and ``Y = i X Z``,
    so every such qubit contributes one factor of i on top of the sign bit.
    """
    return (2 * tableau.phase[row] + _popcount(tableau.x[row] & tableau.z[row])) % 4


def _pauli_mul(x1: int, z1: int, a1: int, x2: int, z2: int,
               a2: int) -> Tuple[int, int, int]:
    """Multiply two Paulis written as ``i**a * prod X**x Z**z``.

    Commuting the left operand's Z's past the right operand's X's is where the
    sign comes from, hence the ``z1 & x2`` parity.
    """
    return x1 ^ x2, z1 ^ z2, (a1 + a2 + 2 * _parity(z1 & x2)) % 4


def _primitive_ops(circuit: Circuit, n: int) -> List[Tuple[str, Tuple[int, ...]]]:
    """Flatten a circuit into tableau primitives, rejecting non-Clifford gates."""
    out: List[Tuple[str, Tuple[int, ...]]] = []
    for gate in circuit.ops:
        name = gate.lowername
        if name in ('measure', 'reset'):
            raise ValueError(f"{name} is not a unitary; a Clifford cannot contain it.")
        if name in _PRIMITIVE and name != 'cx':
            for t in gate.target_iter(n):
                out.append((name, (t,)))
        elif name == 'cx':
            for c, t in gate.control_target_iter(n):
                out.append(('cx', (c, t)))
        elif name in _CLIFFORD_REWRITES:
            rewrite = _CLIFFORD_REWRITES[name]
            if name in ('cz', 'cy', 'swap', 'cnot'):
                pairs = list(gate.control_target_iter(n))
            else:
                pairs = [(t,) for t in gate.target_iter(n)]
            for qubits in pairs:
                out.extend(rewrite(qubits))
        else:
            raise ValueError(
                f"{name} is not a Clifford gate. The Clifford gate set is "
                f"i, x, y, z, h, s, sdg, sx, sxdg, cx, cy, cz, swap.")
    return out


# ------------------------------------------------------------------ sampling

def _symplectic_product(u: Tuple[int, int], v: Tuple[int, int]) -> int:
    return _parity(u[0] & v[1]) ^ _parity(u[1] & v[0])


def _random_combination(span: Sequence[Tuple[int, int]], rng: random.Random) -> Tuple[int, int]:
    x = z = 0
    for vx, vz in span:
        if rng.getrandbits(1):
            x ^= vx
            z ^= vz
    return x, z


def random_clifford(n_qubits: int, seed: Optional[int] = None) -> Clifford:
    """A Clifford drawn uniformly from the `n`-qubit Clifford group (mod phase).

    Uniformity comes from building a random symplectic basis one conjugate pair
    at a time: the image of ``X_i`` is uniform among the non-identity Paulis
    still available, the image of ``Z_i`` is uniform among those anticommuting
    with it, and the rest of the operator is drawn from what commutes with both.
    Counting those choices reproduces ``|Sp(2n, 2)|`` exactly, and the ``2n``
    independent sign bits supply the remaining Pauli factor.

    Pass `seed` for a reproducible draw; it uses its own generator, so it leaves
    the global RNG alone.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be at least 1, got {n_qubits}.")
    rng = random.Random(seed)

    span: List[Tuple[int, int]] = ([(1 << q, 0) for q in range(n_qubits)]
                                   + [(0, 1 << q) for q in range(n_qubits)])
    pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    for _ in range(n_qubits):
        while True:
            v = _random_combination(span, rng)
            if v != (0, 0):
                break
        while True:
            w = _random_combination(span, rng)
            if _symplectic_product(v, w) == 1:
                break
        pairs.append((v, w))

        # What is left must commute with both v and w: project the current span
        # into their symplectic complement, then thin it back to a basis.
        projected = []
        for u in span:
            ux, uz = u
            if _symplectic_product(u, w):
                ux ^= v[0]
                uz ^= v[1]
            if _symplectic_product(u, v):
                ux ^= w[0]
                uz ^= w[1]
            projected.append((ux, uz))
        span = _independent_subset(projected, n_qubits)

    x = [pairs[i][0][0] for i in range(n_qubits)] + [pairs[i][1][0] for i in range(n_qubits)]
    z = [pairs[i][0][1] for i in range(n_qubits)] + [pairs[i][1][1] for i in range(n_qubits)]
    phase = [rng.getrandbits(1) for _ in range(2 * n_qubits)]
    return Clifford(n_qubits, x, z, phase)


def _independent_subset(vectors: Sequence[Tuple[int, int]],
                        n_qubits: int) -> List[Tuple[int, int]]:
    """A maximal linearly independent subset, by Gaussian elimination over F2.

    Vectors are compared as a single ``2n``-bit integer -- the x half in the low
    bits, the z half above it. The width has to be fixed at `n_qubits` rather
    than taken from each vector, or "bit k" would name a different coordinate in
    different rows and the elimination would be meaningless.
    """
    def as_int(vec: Tuple[int, int]) -> int:
        return vec[0] | (vec[1] << n_qubits)

    basis: List[Tuple[int, int]] = []
    pivots: List[Tuple[int, Tuple[int, int]]] = []
    for vec in vectors:
        cur = vec
        for bit, pivot in pivots:
            if (as_int(cur) >> bit) & 1:
                cur = (cur[0] ^ pivot[0], cur[1] ^ pivot[1])
        value = as_int(cur)
        if value:
            pivots.append((value.bit_length() - 1, cur))
            basis.append(cur)
    return basis
