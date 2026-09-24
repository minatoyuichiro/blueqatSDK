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
"""
This module defines Circuit and the setting for circuit.
Modernized for PyTorch Tensor Network backend integration in 2026.
"""

import warnings
from functools import partial, update_wrapper
import typing
from typing import cast, Any, Callable, Dict, Optional, Tuple, Type

import torch

from . import gate
from .gateset import get_op_type, register_operation, unregister_operation
from .typing import CircuitOperation

if typing.TYPE_CHECKING:
    from .gate import Operation
    from .backends.backendbase import Backend
    BackendUnion = typing.Union[None, str, Backend]

GLOBAL_MACROS = {}




def _reverse_index_bits(values: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """Re-index so that qubit 0 is the most-significant bit of the index.

    The values are unchanged; only which index they sit at moves. On a
    symmetric distribution this is the identity, which is exactly why reading
    the convention backwards can go unnoticed.
    """
    if n_qubits <= 1:
        return values
    index = torch.arange(values.shape[0], device=values.device)
    swapped = torch.zeros_like(index)
    for bit in range(n_qubits):
        swapped |= ((index >> bit) & 1) << (n_qubits - 1 - bit)
    return values[swapped]


def _as_pauli_expression(hamiltonian: Any) -> Any:
    """A Pauli expression, or a TypeError saying what one looks like.

    Left as it is when it already is one. A string is parsed. Anything else --
    a dict of coefficients is the common guess -- gets an error naming the
    thing to pass, because the alternative is an AttributeError about
    `simplify` from three frames down, which says nothing about Hamiltonians.
    """
    if hasattr(hamiltonian, 'to_expr'):
        return hamiltonian.to_expr().simplify()
    if hasattr(hamiltonian, 'simplify'):
        return hamiltonian
    if isinstance(hamiltonian, str):
        from .utils import parse_hamiltonian
        return parse_hamiltonian(hamiltonian).simplify()
    raise TypeError(
        f"a Hamiltonian is a Pauli expression, not {type(hamiltonian).__name__}. "
        f"Build one from the operators exported at the top level -- "
        f"`from blueqat import Z, X; Z[0] + 0.5 * X[1]` -- or pass the same "
        f"thing as a string, \"Z[0] + 0.5*X[1]\".")


def _hamiltonian_width(hamiltonian: Any) -> int:
    """How many qubits a Pauli expression names, or 0 if it names none."""
    highest = -1
    for term in getattr(hamiltonian, 'terms', ()):
        for op in getattr(term, 'ops', ()):
            index = getattr(op, 'n', None)
            if index is not None:
                highest = max(highest, int(index))
    return highest + 1


class Circuit:
    """Store the gate operations and call the backends."""
    def __init__(self, n_qubits: int = 0, ops: Optional[list] = None):
        self.ops = ops or []
        self._backends: Dict[str, 'Backend'] = {}
        self.n_qubits = n_qubits

    def __repr__(self):
        return f'Circuit({self.n_qubits}).' + '.'.join(
            str(op) for op in self.ops)

    def __get_backend(self, backend_name):
        from blueqat.backends import BACKENDS
        from blueqat.backends.backendbase import _BACKEND_REGISTRY, get_backend

        try:
            return self._backends[backend_name]
        except KeyError:
            backend = BACKENDS.get(backend_name)
            if backend is not None:
                # インスタンス化してキャッシュ（型でもファクトリlambdaでも呼び出しは同じ）
                self._backends[backend_name] = backend()
                return self._backends[backend_name]
            if backend_name in _BACKEND_REGISTRY:
                # register_backend()経由で登録されたプラグインバックエンド
                self._backends[backend_name] = get_backend(backend_name)
                return self._backends[backend_name]
            raise ValueError(f"Backend {backend_name} doesn't exist.")

    def __backend_runner_wrapper(self, backend_name: str) -> Callable:
        backend = self.__get_backend(backend_name)

        def runner(*args, **kwargs):
            return backend.run(self.ops, self.n_qubits, *args, **kwargs)

        return runner


    def _warn_pending_gates(self) -> None:
        """Say if a gate is still waiting for its qubits when the circuit runs.

        The wrapper reports itself when it is collected, which covers the
        common `c.ry(0.3, 0)` on a line of its own. It does not cover a
        wrapper somebody is still holding -- and a notebook holds the last
        expression of every cell in `_`, which is precisely where this gets
        typed and nothing appears to happen. Running is the other moment the
        question can be asked.
        """
        pending = getattr(self, '_pending_gates', None)
        if not pending:
            return
        waiting = [w for w in pending if not getattr(w, '_applied', True)]
        if not waiting:
            return
        import warnings
        for wrapper in waiting:
            wrapper._applied = True          # reported once, not twice
            warnings.warn(
                f"running a circuit while `.{wrapper.op_type.lowername}` is "
                f"still waiting for its qubits, so that gate is not in it. "
                f"Write `circuit.{wrapper._suggestion()}`.",
                SyntaxWarning, stacklevel=3)

    def __getattr__(self, name: str) -> CircuitOperation[Any]:
        op_type = get_op_type(name)
        if op_type:
            wrapper = _GateWrapper(self, op_type)
            # Held weakly, so that tracking a wrapper cannot keep it alive.
            # `run` asks whether any are still waiting for their qubits: a
            # wrapper that is discarded reports itself when collected, but one
            # the caller happens to be holding does not, and a notebook holds
            # the last expression of every cell in `_`. That is exactly where
            # somebody types `c.ry(0.3, 0)` and sees nothing happen.
            try:
                self._pending_gates.add(wrapper)
            except AttributeError:
                import weakref
                object.__setattr__(self, '_pending_gates', weakref.WeakSet())
                self._pending_gates.add(wrapper)
            return wrapper
        if name in GLOBAL_MACROS:
            macro = update_wrapper(partial(GLOBAL_MACROS[name], self), GLOBAL_MACROS[name])
            return cast(CircuitOperation[Any], macro)
        if name.startswith("run_with_"):
            # メソッド内部で遅延インポート
            from blueqat.backends import BACKENDS
            from blueqat.backends.backendbase import _BACKEND_REGISTRY
            backend_name = name[9:]
            if backend_name in BACKENDS or backend_name in _BACKEND_REGISTRY:
                return self.__backend_runner_wrapper(backend_name)
            raise AttributeError(f"Backend '{backend_name}' does not exist.")
        raise AttributeError(
            f"'Circuit' object has no attribute or gate '{name}'")

    def __add__(self, other: 'Circuit') -> 'Circuit':
        if not isinstance(other, Circuit):
            return NotImplemented
        c = self.copy()
        c += other
        return c

    def __iadd__(self, other: 'Circuit') -> 'Circuit':
        if not isinstance(other, Circuit):
            return NotImplemented
        self.ops += other.ops
        self.n_qubits = max(self.n_qubits, other.n_qubits)
        return self

    def copy(self, copy_backends: bool = True) -> 'Circuit':
        """Copy the circuit."""
        copied = Circuit(self.n_qubits, self.ops.copy())
        if copy_backends:
            copied._backends = {k: v.copy() for k, v in self._backends.items()}
        return copied

    def dagger(self, ignore_measurement: bool = False) -> 'Circuit':
        """Make Hermitian conjugate of the circuit.

        If the circuit contains measurement or reset (which have no Hermitian
        conjugate), ValueError is raised, unless `ignore_measurement` is True,
        in which case those operations are simply dropped."""
        ops = []
        for g in reversed(self.ops):
            if not hasattr(g, 'dagger'):
                if ignore_measurement:
                    continue
                raise ValueError(
                    'Cannot make the Hermitian conjugate of this circuit because '
                    f'the circuit contains a non-invertible operation `{g.lowername}`.')
            ops.append(g.dagger())

        copied = Circuit(self.n_qubits, ops)
        return copied

    def run(self, backend: Optional[str] = None, *args, **kwargs) -> Any:
        """Run the circuit. Passes parameters to the PyTorch-based backend.

        Beyond the backend's own arguments (``shots``, ``returns``, ``mode``,
        ``hamiltonian``, ``amplitude``, ``initial``, ...), two arguments shape
        sampled results:

        ``seed``
            Fix every random draw of this run -- shot sampling, mid-circuit
            collapse and large-``n`` perfect sampling -- so that the same
            circuit and seed give the same counts. It drives a private
            ``torch.Generator``, leaving the global RNG untouched.
        ``bit_order``
            Layout of the counts keys: ``'q0_last'`` (the default, and
            blueqat's long-standing order, where ``key[-1]`` is qubit 0) or
            ``'q0_first'``, where ``key[i]`` is qubit i, as cloud APIs report
            it. Keys are zero-padded to ``n_qubits`` in either order.
        """
        return self._resolve_backend(backend, kwargs).run(
            self.ops, self.n_qubits, *args, **kwargs)

    def _resolve_backend(self, backend: 'BackendUnion', kwargs: dict) -> 'Backend':
        """The backend a call should go to, given what it was asked for.

        Noise needs a density matrix, which the default backends do not carry, so
        asking for noise is also a choice of backend. Quasi-static noise counts
        too: its result is an average over frozen detunings, which is a mixture.

        Every entry point routes through here. When only `run` did, the same
        request written as `shots(...)` or `probs(...)` reached a backend that
        does not know the argument and dropped it, returning a noiseless answer
        with no error -- the failure this exists to prevent.
        """
        # Every entry point -- run, statevector, probs, shots, expect --
        # comes through here, so asking once here cannot be bypassed by a new
        # one being added later.
        self._warn_pending_gates()
        from blueqat.backends import DEFAULT_BACKEND_NAME

        if backend is not None:
            return self.__get_backend(backend) if isinstance(backend, str) else backend
        noisy = (kwargs.get('noise') is not None
                 or kwargs.get('quasi_static') is not None
                 or kwargs.get('noise_scale') is not None)
        return self.__get_backend('density' if noisy else DEFAULT_BACKEND_NAME)

    def to_qasm(self, output_prologue: bool = True) -> str:
        """Convert this circuit into an OpenQASM 2.0 program string."""
        from blueqat.backends.qasm_output_backend import QasmOutputBackend
        return QasmOutputBackend().run(self.ops, self.n_qubits, output_prologue=output_prologue)

    def statevector(self, backend: 'BackendUnion' = None,
                    bit_order: str = 'q0_last', **kwargs) -> torch.Tensor:
        """Run the circuit and get a statevector as a PyTorch Tensor to keep
        gradients intact.

        Amplitude ``v[k]`` belongs to the basis state whose qubit ``q`` is bit
        ``q`` of ``k`` -- qubit 0 is the *least*-significant bit of the index,
        the same convention as everywhere else in the SDK and as
        `blueqat.BIT_ORDER` reports. So for two qubits the order is |00>, |01>
        with qubit 0 set, |10> with qubit 1 set, |11>.

        Reading it the other way round is a mistake nothing catches: it gives
        the mirror image of the answer, and on a symmetric state -- a GHZ, a W,
        anything permutation-invariant -- the two agree, so it can go unnoticed
        through a whole set of examples and fail on the one that is not
        symmetric.

        `bit_order='q0_first'` puts qubit 0 in the most-significant bit
        instead, as some other toolkits do. It is the same argument name and
        the same values that `run(shots=...)` and `probs()` take. Having it on
        some of the three and not the others is itself the trap, because then
        "I checked with run()" stops being an answer about the other two --
        and before this it was worse than absent here: the argument was
        accepted, validated, and then silently ignored, so asking for the
        other convention returned the default one with nothing said."""
        if kwargs.get('returns'):
            raise ValueError('Circuit.statevector has no argument `returns`.')
        # Imported here rather than at module scope: backendbase pulls in the
        # backends package, which imports this module back.
        from .backends.backendbase import BIT_ORDERS
        if bit_order not in BIT_ORDERS:
            raise ValueError(
                f"bit_order must be one of {BIT_ORDERS}, got {bit_order!r}.")
        backend = self._resolve_backend(backend, kwargs)

        if hasattr(backend, 'statevector'):
            state = backend.statevector(self.ops, self.n_qubits, **kwargs)
        else:
            state = backend.run(self.ops, self.n_qubits, returns='statevector',
                                **kwargs)
        if bit_order == 'q0_first':
            return _reverse_index_bits(state, self.n_qubits)
        return state

    def shots(self, shots: int, backend: 'BackendUnion' = None, **kwargs) -> typing.Counter[str]:
        """Run the circuit and get shot counts as a result.

        Accepts the same ``seed`` and ``bit_order`` arguments as
        :meth:`~blueqat.circuit.Circuit.run`."""
        if kwargs.get('returns'):
            raise ValueError('Circuit.shots has no argument `returns`.')
        backend = self._resolve_backend(backend, kwargs)

        if hasattr(backend, 'shots'):
            return backend.shots(self.ops, self.n_qubits, shots=shots, **kwargs)
        return backend.run(self.ops, self.n_qubits, shots=shots, returns='shots', **kwargs)

    def oneshot(self, backend: 'BackendUnion' = None, **kwargs) -> Tuple[torch.Tensor, str]:
        """Run the circuit once and return the post-measurement statevector together
        with the single measured bitstring."""
        if kwargs.get('returns'):
            raise ValueError('Circuit.oneshot has no argument `returns`.')
        backend = self._resolve_backend(backend, kwargs)
        vec, cnt = backend.run(self.ops, self.n_qubits, shots=1, returns='statevector_and_shots', **kwargs)
        return vec, next(iter(cnt))

    def _expanded_applications(self, ops: Optional[list] = None):
        """Yield (lowername, qubit-tuple) for each atomic gate application,
        expanding slices/multi-targets (and recursing into named blocks) the
        same way the backends do."""
        from .gate import (Barrier, Gate, GateBlock, Measurement, OneQubitGate,
                           Reset, TwoQubitGate)
        n_qubits = self.n_qubits
        for op in (self.ops if ops is None else ops):
            if isinstance(op, GateBlock):
                yield from self._expanded_applications(op.ops)
            elif isinstance(op, Barrier):
                yield op.lowername, tuple(op.target_iter(n_qubits))
            elif isinstance(op, (OneQubitGate, Measurement, Reset)):
                for t in op.target_iter(n_qubits):
                    yield op.lowername, (t, )
            elif isinstance(op, TwoQubitGate):
                for c, t in op.control_target_iter(n_qubits):
                    yield op.lowername, (c, t)
            elif isinstance(op, Gate):
                yield op.lowername, tuple(op.targets)
            else:
                yield op.lowername, tuple(op.target_iter(n_qubits))

    def depth(self) -> int:
        """Circuit depth: length of the longest gate sequence on any qubit path,
        counting each expanded gate application (as in Qiskit). Barriers don't
        add depth."""
        depths = [0] * self.n_qubits
        for name, qubits in self._expanded_applications():
            if name == 'barrier' or not qubits:
                continue
            d = max(depths[q] for q in qubits) + 1
            for q in qubits:
                depths[q] = d
        return max(depths, default=0)

    def count_ops(self) -> typing.Counter[str]:
        """Count expanded gate applications by name (as in Qiskit's count_ops)."""
        import collections
        return collections.Counter(name for name, _ in self._expanded_applications())

    def probs(self, qubits: Optional[typing.Sequence[int]] = None,
              backend: 'BackendUnion' = None, bit_order: str = 'q0_last',
              **kwargs) -> torch.Tensor:
        """Measurement probabilities of the circuit's final state, optionally
        marginalized onto `qubits` (as in PennyLane's `qml.probs`).

        Returns a tensor of length 2**len(qubits). By default index bit j is
        the outcome of `qubits[j]`: the first listed qubit is the
        least-significant bit of the index, matching the SDK-wide convention
        and what `blueqat.BIT_ORDER` reports. Differentiable.

        `bit_order='q0_first'` reverses that, putting the first listed qubit in
        the most-significant bit, which is what some other toolkits do. The
        name and the values are the same ones `run(shots=...)` takes, so the
        two do not have to be remembered separately.

        Under noise there is no statevector to square; the probabilities are the
        density matrix's diagonal, and are read from there."""
        from .backends.backendbase import BIT_ORDERS
        if bit_order not in BIT_ORDERS:
            raise ValueError(
                f"bit_order must be one of {BIT_ORDERS}, got {bit_order!r}.")
        resolved = self._resolve_backend(backend, kwargs)
        if getattr(resolved, 'returns_density_matrix', False):
            rho = resolved.run(self.ops, self.n_qubits, **kwargs)
            p = torch.diagonal(rho).real.clone()
        else:
            p = torch.abs(self.statevector(backend, **kwargs)) ** 2
        if qubits is None:
            return _reverse_index_bits(p, self.n_qubits) if bit_order == 'q0_first' else p
        keep = list(qubits)
        if len(set(keep)) != len(keep):
            raise ValueError('qubits must not contain duplicates.')
        n = self.n_qubits
        if any(not 0 <= q < n for q in keep):
            raise ValueError(f'qubits must be in range(0, {n}).')
        # After reshape, axis k corresponds to qubit n-1-k (the statevector
        # index has qubit 0 as its least-significant bit).
        t = p.reshape((2, ) * n)
        keep_set = set(keep)
        sum_axes = [n - 1 - q for q in range(n) if q not in keep_set]
        if sum_axes:
            t = t.sum(dim=sum_axes)
        remaining = [q for q in reversed(range(n)) if q in keep_set]
        # reshape(-1) makes the first axis most significant, so order axes as
        # [last listed qubit, ..., first listed qubit].
        t = t.permute([remaining.index(q) for q in reversed(keep)])
        marginal = t.reshape(-1)
        if bit_order == 'q0_first':
            marginal = _reverse_index_bits(marginal, len(keep))
        return marginal

    def expect(self, hamiltonian: Any, backend: 'BackendUnion' = None, **kwargs) -> torch.Tensor:
        """Expectation value <psi|H|psi> of a Pauli-expression Hamiltonian on
        the circuit's final state. Differentiable.

        A Hamiltonian is a Pauli expression -- ``Z[0] + 0.5 * X[1]``, built from
        the operators exported at the top level -- or a string
        (``"Z[0] + 0.5*X[1]"``). A dict of coefficients is not one, and neither
        is a matrix.

        The circuit may be narrower than the Hamiltonian: qubits the circuit
        never mentions are in |0>, which is a perfectly good state to take an
        expectation in, and ``Circuit().expect(Z[0]) == 1`` rather than an
        error about a zero-qubit state.
        """
        hamiltonian = _as_pauli_expression(hamiltonian)
        width = _hamiltonian_width(hamiltonian)
        if width > self.n_qubits:
            # Widen a copy rather than this circuit: expect() is a question,
            # and asking it should not change the thing being asked about.
            widened = Circuit(width, list(self.ops))
            return widened.run(backend, hamiltonian=hamiltonian, **kwargs)
        return self.run(backend, hamiltonian=hamiltonian, **kwargs)

    def exp_pauli(self, paulis: typing.Mapping[int, str], theta: Any) -> 'Circuit':
        """Append ``exp(-i * theta * P)``, the time evolution of a single Pauli product.

        `paulis` maps a qubit index to its Pauli letter, so the operator is stated
        without reference to any bit order or overall width::

            Circuit().exp_pauli({0: 'X', 1: 'X', 2: 'Z', 3: 'Y'}, 0.3)  # exp(-0.3i XXZY)

        Since ``P**2 == I``, this is exactly ``cos(theta) - i sin(theta) P``. The
        convention (no factor of 1/2) matches
        :meth:`~blueqat.utils.Term.get_time_evolution`; note that a single-qubit
        ``{q: 'Z'}`` is therefore ``rz(2 * theta)[q]``.

        `theta` may be a ``torch.Tensor``, in which case the gradient flows through.
        Letters are case-insensitive, and ``'I'`` entries are ignored. A product of
        nothing but identities is a global phase, which a statevector does not carry,
        so it appends no gates.
        """
        ops = []
        for qubit, letter in paulis.items():
            if not isinstance(qubit, int) or isinstance(qubit, bool) or qubit < 0:
                raise ValueError(f"Qubit index must be a non-negative int, got {qubit!r}.")
            letter = str(letter).upper()
            if letter not in ('X', 'Y', 'Z', 'I'):
                raise ValueError(f"Pauli letter must be one of X, Y, Z, I, got {letter!r}.")
            if letter != 'I':
                ops.append((qubit, letter))
        if not ops:
            return self
        ops.sort()

        half_pi = torch.pi / 2
        # Rotate each factor into the Z basis (H X H = Z, RX(+pi/2) Y RX(-pi/2) = Z),
        # accumulate the parity of the whole product onto the last qubit, rotate it by
        # rz(2*theta), then undo both. Same construction as Term.get_time_evolution.
        for qubit, letter in ops:
            if letter == 'X': self.h[qubit]
            elif letter == 'Y': self.rx(half_pi)[qubit]
        for i in range(1, len(ops)):
            self.cx[ops[i - 1][0], ops[i][0]]
        self.rz(2 * theta)[ops[-1][0]]
        for i in range(len(ops) - 1, 0, -1):
            self.cx[ops[i - 1][0], ops[i][0]]
        for qubit, letter in ops:
            if letter == 'X': self.h[qubit]
            elif letter == 'Y': self.rx(-half_pi)[qubit]
        return self

    def block(self, name: str) -> '_BlockContext':
        """Group the operations appended inside the `with` body into a named,
        nestable block (as in the sub-circuits of Shor's algorithm)::

            c = Circuit(4)
            with c.block("QFT"):
                c.h[0].cphase(math.pi / 2)[0, 1]
                ...

        Blocks change nothing about execution -- every backend transparently
        sees the inner gates -- but the structure is kept in `repr()`,
        `Circuit.tree()`, and survives `dagger()` (as a mirrored block named
        `name + '†'`)."""
        return _BlockContext(self, name)

    def append_block(self, name: str, subcircuit: 'Circuit',
                     offset: int = 0) -> 'Circuit':
        """Append an existing circuit as a named block.

        `offset` shifts every qubit index of `subcircuit`, so a library
        circuit built on qubits 0..k can be placed anywhere. Shifting
        resolves slice targets against `subcircuit.n_qubits` and preserves
        any nested block structure inside `subcircuit`."""
        from .circuit_funcs.flatten import flatten
        from .gate import GateBlock
        if offset < 0:
            raise ValueError('offset must not be negative.')

        n_sub = subcircuit.n_qubits

        def _shift_ops(ops: list) -> list:
            out = []
            for op in ops:
                if isinstance(op, GateBlock):
                    out.append(GateBlock(op.name, _shift_ops(op.ops)))
                    continue
                # flatten a single op to resolve slices into explicit targets
                for atom in flatten(Circuit(n_sub, [op])).ops:
                    targets = atom.targets
                    if isinstance(targets, int):
                        shifted: Any = targets + offset
                    else:
                        shifted = tuple(t + offset for t in targets)
                    options = None
                    if getattr(atom, 'key', None) is not None:
                        options = {'key': atom.key}
                        if atom.duplicated is not None:
                            options['duplicated'] = atom.duplicated
                    out.append(atom.create(shifted, atom.params, options))
            return out

        if offset == 0:
            ops = [op for op in subcircuit.ops]
        else:
            ops = _shift_ops(subcircuit.ops)
        width = n_sub + offset
        self.ops.append(GateBlock(name, ops))
        self.n_qubits = max(self.n_qubits, width)
        return self

    def tree(self) -> str:
        """A text rendering of the circuit's nested block structure::

            Circuit(4)
            ├─ h[0]
            └─ QFT
               ├─ cphase(1.5708)[0, 1]
               └─ ...

        Blocks appear by name with their contents beneath them; plain gates
        outside any block are listed at the top level."""
        from .gate import GateBlock

        def _lines(ops, prefix: str):
            out = []
            for i, op in enumerate(ops):
                last = i == len(ops) - 1
                branch = '└─ ' if last else '├─ '
                cont = '   ' if last else '│  '
                if isinstance(op, GateBlock):
                    out.append(f'{prefix}{branch}{op.name}')
                    out.extend(_lines(op.ops, prefix + cont))
                else:
                    out.append(f'{prefix}{branch}{op}')
            return out

        return '\n'.join([f'Circuit({self.n_qubits})'] + _lines(self.ops, ''))

    def ancilla(self, n: int = 1, pos: Optional[int] = None, stop: Optional[int] = None,
                reset: bool = True) -> '_AncillaContext':
        """Context manager allocating temporary ancilla qubit(s) for use inside the `with` block.

        By default, appends `n` fresh qubits past the circuit's current width:

            with c.ancilla() as a:
                c.cx[0, a[0]]

        `pos`/`stop` instead pin the ancilla range to specific qubit indices
        (`range(pos, stop)`; `stop` defaults to `pos + n`):

            with c.ancilla(pos=4, stop=6, reset=True) as a:
                c.cx[3, a[0]]

        If `reset` is true (the default), a `reset` gate is appended for each
        ancilla qubit on exiting the block, so they're back at ``|0>`` and safe to
        reuse elsewhere in the circuit.
        """
        if pos is not None:
            # A negative pos resolves later as an index from the end, so an
            # ancilla silently lands on a data qubit -- and the automatic reset
            # on leaving the block erases it. `append_block` already refuses
            # negative offsets; this is the same rule.
            if pos < 0:
                raise ValueError(f"ancilla pos must be non-negative, got {pos}.")
            if stop is not None and stop < pos:
                raise ValueError(f"ancilla stop ({stop}) must not be below pos ({pos}).")
            indices = list(range(pos, stop if stop is not None else pos + n))
            self.n_qubits = max(self.n_qubits, (max(indices) + 1) if indices else 0)
        else:
            indices = list(range(self.n_qubits, self.n_qubits + n))
            self.n_qubits += n
        return _AncillaContext(self, indices, reset)


class _BlockContext:
    """Context manager returned by `Circuit.block()`. On exit, the operations
    appended inside the body are wrapped into a single named GateBlock
    (supports nesting: an inner block closes before its enclosing one)."""
    def __init__(self, circuit: Circuit, name: str) -> None:
        self.circuit = circuit
        self.name = name
        self._start = 0

    def __enter__(self) -> '_BlockContext':
        self._start = len(self.circuit.ops)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            return
        from .gate import GateBlock
        inner = self.circuit.ops[self._start:]
        del self.circuit.ops[self._start:]
        self.circuit.ops.append(GateBlock(self.name, inner))


class _AncillaContext:
    """Context manager returned by `Circuit.ancilla()`. See that method's docstring."""
    def __init__(self, circuit: Circuit, indices: list, reset: bool) -> None:
        self.circuit = circuit
        self.indices = indices
        self.reset = reset

    def __getitem__(self, i: int) -> int:
        return self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __enter__(self) -> '_AncillaContext':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.reset and exc_type is None:
            for idx in self.indices:
                self.circuit.reset[idx]


class _GateWrapper(CircuitOperation[Circuit]):
    """A gate waiting to be told which qubits it acts on.

    `circuit.ry(theta)` produces one of these; `[0]` is what actually appends
    the gate. Forgetting the subscript therefore adds nothing, and the circuit
    runs and returns a perfectly ordinary answer for the gates that were
    added -- which reads as "the gate had no effect" and has been reported as
    exactly that. So an unsubscripted wrapper says so when it is discarded.
    """

    def __init__(self, circuit: Circuit, op_type: Type['Operation']):
        self.circuit = circuit
        self.op_type = op_type
        self.params = ()
        self.options = None
        self._applied = False

    def __call__(self, *args, **kwargs) -> '_GateWrapper':
        self.params = args
        if kwargs:
            self.options = kwargs
        return self

    def __getitem__(self, targets) -> 'Circuit':
        self._applied = True
        self.circuit.ops.append(
            self.op_type.create(targets, self.params, self.options))
        self.circuit.n_qubits = max(
            gate.get_maximum_index(targets) + 1, self.circuit.n_qubits)
        return self.circuit

    def __del__(self) -> None:
        # Nothing was appended, so the circuit is quietly missing a gate. The
        # alternative to saying so is a run that succeeds and answers as
        # though the gate were not there, which is indistinguishable from the
        # gate having no effect.
        if getattr(self, '_applied', True):
            return
        try:
            import warnings
            warnings.warn(
                f"`.{self.op_type.lowername}` was written without saying which "
                f"qubits it acts on, so no gate was added. Write "
                f"`circuit.{self._suggestion()}`. Several other toolkits pass "
                f"the qubit as an argument; blueqat spells it as a subscript, "
                f"and the difference is silent -- the circuit runs and gives "
                f"the answer for a circuit without this gate.",
                SyntaxWarning, stacklevel=2)
        except Exception:                                # pragma: no cover
            pass

    def _suggestion(self) -> str:
        """The call the author probably meant, built from what they wrote.

        A warning that names the mistake is worth less than one that spells
        the fix, and here the fix can be derived: whatever trailing integers
        were passed are almost certainly the qubits.
        """
        name = self.op_type.lowername
        params = list(self.params)
        qubits = []
        while params and isinstance(params[-1], int) and not isinstance(params[-1], bool):
            qubits.insert(0, params.pop())
        if not qubits:
            qubits = [0]
        subscript = ', '.join(str(q) for q in qubits)
        if params:
            return f"{name}({', '.join(repr(p) for p in params)})[{subscript}]"
        return f"{name}[{subscript}]"

    def __str__(self) -> str:
        args_str = str(self.params) if self.params else ""
        if self.options:
            args_str += str(self.options)
        return self.op_type.lowername + args_str


class BlueqatGlobalSetting:
    """Setting for Blueqat."""
    @staticmethod
    def register_macro(name: str, func: Callable, allow_overwrite: bool = False) -> None:
        """Register new macro to Circuit."""
        if hasattr(Circuit, name):
            if allow_overwrite:
                warnings.warn(f"Circuit has attribute `{name}`.")
            else:
                raise ValueError(f"Circuit has attribute `{name}`.")
        if name.startswith("run_with_"):
            if allow_overwrite:
                warnings.warn(f"Gate name `{name}` may conflict with run of backend.")
            else:
                raise ValueError(f"Gate name `{name}` shall not start with 'run_with_'.")
        if not allow_overwrite:
            if get_op_type(name) is not None:
                raise ValueError(f"Gate '{name}' already exists in gate set.")
            if name in GLOBAL_MACROS:
                raise ValueError(f"Macro '{name}' already exists.")
        GLOBAL_MACROS[name] = func

    @staticmethod
    def unregister_macro(name: str) -> None:
        """Unregister a macro."""
        if name not in GLOBAL_MACROS:
            raise ValueError(f"Macro '{name}' is not registered.")
        del GLOBAL_MACROS[name]

    @staticmethod
    def register_gate(name: str, gateclass: Type['Operation'], allow_overwrite: bool = False) -> None:
        """Register new gate to gate set."""
        if hasattr(Circuit, name):
            if allow_overwrite:
                warnings.warn(f"Circuit has attribute `{name}`.")
            else:
                raise ValueError(f"Circuit has attribute `{name}`.")
        if name.startswith("run_with_"):
            if allow_overwrite:
                warnings.warn(f"Gate name `{name}` may conflict with run of backend.")
            else:
                raise ValueError(f"Gate name `{name}` shall not start with 'run_with_'.")
        if not allow_overwrite:
            if get_op_type(name) is not None:
                raise ValueError(f"Gate '{name}' already exists in gate set.")
            if name in GLOBAL_MACROS:
                raise ValueError(f"Macro '{name}' already exists.")
        register_operation(name, gateclass)

    @staticmethod
    def unregister_gate(name: str) -> None:
        """Unregister a gate from gate set."""
        if get_op_type(name) is None:
            raise ValueError(f"Gate '{name}' is not registered.")
        unregister_operation(name)

    @staticmethod
    def register_backend(name: str, backend: Type['Backend'], allow_overwrite: bool = False) -> None:
        """Register new backend."""
        from blueqat.backends import BACKENDS
        if hasattr(Circuit, "run_with_" + name):
            if allow_overwrite:
                warnings.warn(f"Circuit has attribute `run_with_{name}`.")
            else:
                raise ValueError(f"Circuit has attribute `run_with_{name}`.")
        if not allow_overwrite and name in BACKENDS:
            raise ValueError(f"Backend '{name}' is already registered.")
        BACKENDS[name] = backend

    @staticmethod
    def unregister_backend(name: str) -> None:
        """Unregister a backend."""
        from blueqat.backends import BACKENDS
        if name not in BACKENDS:
            raise ValueError(f"Backend '{name}' is not registered.")
        del BACKENDS[name]

    @staticmethod
    def set_default_backend(name: str) -> None:
        """Set the default backend to be used by `Circuit`."""
        from blueqat.backends import BACKENDS
        if name not in BACKENDS:
            raise ValueError(f"Backend '{name}' is not registered.")
        # モジュール参照経由でグローバル変数を書き換える
        import blueqat.backends
        blueqat.backends.DEFAULT_BACKEND_NAME = name

    @staticmethod
    def get_default_backend_name() -> str:
        """Get the default backend name."""
        from blueqat.backends import DEFAULT_BACKEND_NAME
        return DEFAULT_BACKEND_NAME