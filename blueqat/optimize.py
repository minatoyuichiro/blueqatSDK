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
"""Peephole circuit optimization.

Three rewrites, applied to a fixed point: drop gates that are the identity,
cancel adjacent inverses, and merge adjacent rotations about the same axis.
"Adjacent" means adjacent *on the qubits involved* -- gates in between that
touch nothing in common do not block a cancellation::

    Circuit(2).h[0].x[1].h[0]        ->  Circuit(2).x[1]

Every rewrite preserves the unitary exactly, global phase included: a rotation
is dropped only at a multiple of its true identity period (4*pi for ``rx``,
``ry``, ``rz`` and the two-qubit Pauli rotations; 2*pi for ``p`` and ``exch``),
never at ``2*pi`` where those gates equal ``-I``.

Barriers, measurements and resets are never removed and never optimized
across, so they keep meaning what they meant.
"""

import math
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple

import torch

from .circuit import Circuit

__all__ = ['optimize', 'drop_identities', 'cancel_inverses', 'merge_rotations']

_TWO_PI = 2.0 * math.pi
_FOUR_PI = 4.0 * math.pi

#: Angle at which each rotation is *exactly* the identity, global phase included.
#: Measured rather than assumed: rz(2*pi) is -I, not I, so halving these would
#: silently flip the sign of a statevector.
IDENTITY_PERIOD = {
    'rx': _FOUR_PI, 'ry': _FOUR_PI, 'rz': _FOUR_PI,
    'rxx': _FOUR_PI, 'ryy': _FOUR_PI, 'rzz': _FOUR_PI,
    'crx': _FOUR_PI, 'cry': _FOUR_PI, 'crz': _FOUR_PI,
    'phase': _TWO_PI, 'cphase': _TWO_PI, 'exch': _TWO_PI,
}

#: Gates that are their own inverse.
SELF_INVERSE = frozenset({
    'h', 'x', 'y', 'z', 'cx', 'cy', 'cz', 'swap', 'ccx', 'ccz', 'cswap',
})

#: Gates that cancel against a different gate.
INVERSE_PAIRS = {
    's': 'sdg', 'sdg': 's', 't': 'tdg', 'tdg': 't',
    'sx': 'sxdg', 'sxdg': 'sx', 'iswap': 'iswapdg', 'iswapdg': 'iswap',
    'zz': 'zzdg', 'zzdg': 'zz',
}

#: Gates unchanged by swapping their two targets, so ``cz[0, 1]`` and
#: ``cz[1, 0]`` are the same gate and may be matched against each other.
#: Measured, not assumed: ``cz`` and ``cp`` are symmetric but ``crz`` is not,
#: since it phases ``|10>`` and ``|11>`` differently.
SYMMETRIC_TARGETS = frozenset({
    'cz', 'swap', 'rxx', 'ryy', 'rzz', 'zz', 'zzdg', 'cphase', 'exch',
    'iswap', 'iswapdg',
})

#: Nothing is reordered across these, and they are never removed.
_OPAQUE = frozenset({'barrier', 'measure', 'reset'})


def _qubits(op: Any) -> Tuple[int, ...]:
    targets = op.targets
    if isinstance(targets, int):
        return (targets, )
    return tuple(int(t) for t in targets)


def _same_targets(a: Any, b: Any) -> bool:
    """Whether two operations act on the same qubits in a way that lets them be
    matched -- as an ordered pair, unless the gate is symmetric in its targets."""
    qa, qb = _qubits(a), _qubits(b)
    if a.lowername in SYMMETRIC_TARGETS:
        return sorted(qa) == sorted(qb)
    return qa == qb


def _is_trainable(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.requires_grad


def _next_overlapping(ops: Sequence[Any], start: int,
                      qubits: Set[int]) -> Optional[int]:
    """Index of the next operation sharing a qubit with `qubits`, or None."""
    for j in range(start, len(ops)):
        if qubits & set(_qubits(ops[j])):
            return j
    return None


def _rebuild(op: Any, params: tuple) -> Any:
    options = None
    if getattr(op, 'key', None) is not None:
        options = {'key': op.key}
        if getattr(op, 'duplicated', None) is not None:
            options['duplicated'] = op.duplicated
    return type(op).create(op.targets, params, options)


def drop_identities(ops: List[Any]) -> Tuple[List[Any], bool]:
    """Remove ``i`` gates and rotations sitting at a multiple of their period."""
    out: List[Any] = []
    changed = False
    for op in ops:
        name = op.lowername
        if name == 'i':
            changed = True
            continue
        period = IDENTITY_PERIOD.get(name)
        if period is not None and op.params:
            angle = op.params[0]
            # A zero-valued *trainable* angle is not a removable gate: the
            # gradient with respect to it is generally nonzero, so deleting it
            # would change what the optimizer sees, not just the circuit.
            if not _is_trainable(angle):
                value = float(angle) % period
                if min(value, period - value) < 1e-12:
                    changed = True
                    continue
        out.append(op)
    return out, changed


def cancel_inverses(ops: List[Any]) -> Tuple[List[Any], bool]:
    """Delete pairs of adjacent operations that undo each other."""
    ops = list(ops)
    removed = set()
    changed = False
    for i, op in enumerate(ops):
        if i in removed or op.lowername in _OPAQUE:
            continue
        if op.lowername in SELF_INVERSE:
            wanted = op.lowername
        elif op.lowername in INVERSE_PAIRS:
            wanted = INVERSE_PAIRS[op.lowername]
        else:
            continue
        qubits = set(_qubits(op))
        j = _next_overlapping(ops, i + 1, qubits)
        while j is not None and j in removed:
            j = _next_overlapping(ops, j + 1, qubits)
        if j is None:
            continue
        other = ops[j]
        if other.lowername == wanted and _same_targets(op, other):
            removed.update((i, j))
            changed = True
    return [op for i, op in enumerate(ops) if i not in removed], changed


def merge_rotations(ops: List[Any]) -> Tuple[List[Any], bool]:
    """Fuse adjacent rotations about the same axis on the same qubits."""
    ops = list(ops)
    removed = set()
    changed = False
    for i, op in enumerate(ops):
        if i in removed or op.lowername not in IDENTITY_PERIOD or not op.params:
            continue
        qubits = set(_qubits(op))
        j = _next_overlapping(ops, i + 1, qubits)
        while j is not None and j in removed:
            j = _next_overlapping(ops, j + 1, qubits)
        if j is None:
            continue
        other = ops[j]
        if other.lowername != op.lowername or not _same_targets(op, other):
            continue
        # Every rotation here is additive in its angle, so the pair becomes one
        # gate. Tensor angles stay tensors, keeping the parameter differentiable.
        ops[i] = _rebuild(op, (op.params[0] + other.params[0], ))
        removed.add(j)
        changed = True
    return [op for i, op in enumerate(ops) if i not in removed], changed


DEFAULT_PASSES: Tuple[Callable[[List[Any]], Tuple[List[Any], bool]], ...] = (
    merge_rotations, cancel_inverses, drop_identities,
)


def optimize(circuit: Circuit,
             passes: Optional[Sequence[Callable]] = None,
             max_rounds: int = 20) -> Circuit:
    """Return an equivalent circuit with the peephole rewrites applied.

    The circuit is flattened first, so named blocks and sliced targets become
    explicit single-gate applications -- rewriting needs to see the individual
    applications, and a merge across a block boundary would have no structure
    left to belong to anyway.
    """
    from .circuit_funcs.flatten import flatten

    ops = list(flatten(circuit).ops)
    active = DEFAULT_PASSES if passes is None else tuple(passes)
    for _ in range(max_rounds):
        changed_any = False
        for step in active:
            ops, changed = step(ops)
            changed_any = changed_any or changed
        if not changed_any:
            break
    return Circuit(circuit.n_qubits, ops)
