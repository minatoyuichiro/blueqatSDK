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
"""Statevector simulation on an NVIDIA GPU, through cuQuantum's cuStateVec.

    import blueqat.backends.cuquantum_backend    # registers 'cuquantum'
    Circuit(24).h[0].cx[0, 1].run(backend='cuquantum', shots=1000)

The work is in two layers, and the split is deliberate.

`_apply_all` decides *what* to do: which matrix, on which qubits, in what
order, and what to do about a measurement. It is pure, and a test runs it
against a plain-torch applier and requires the result to equal `TorchBackend`'s
statevector to machine precision. So everything about the translation --
qubit order, which end is qubit 0, gate definitions -- is checked on a machine
with no GPU at all.

`_CuStateVec` decides *how*: it is the cuStateVec calls themselves and nothing
else. That part cannot be exercised here, and says so rather than implying
otherwise.

Gate matrices come from `gate.matrix()`, the same method the CPU backends use,
and the whole operator goes down at once: `CXGate.matrix()` is the 4x4, not X,
so there is no separate notion of a control here. Rewriting the matrices would
be the obvious way for a connector to drift away from the simulator it is
meant to agree with.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..gate import IFallbackOperation, Measurement, Operation
from .backendbase import Backend, apply_bit_order, register_backend

#: cuStateVec indexes qubits with 0 as the least-significant bit of the state
#: index, which is also blueqat's convention. Stated as a constant rather than
#: relied on quietly: if it were the other way round every result would be the
#: mirror image, with nothing to indicate it -- the failure blueqat.BIT_ORDER
#: exists to make visible.
QUBIT_ZERO_IS_LEAST_SIGNIFICANT = True


class CuQuantumNotAvailable(RuntimeError):
    """cuQuantum could not be used, with the reason.

    Raised instead of an ImportError so the message can say which of the
    several requirements is missing, and that none of them are needed for the
    CPU backends.
    """


def _require_cuquantum() -> Any:
    """The cuStateVec module, or an error naming what is missing."""
    missing = []
    try:
        import torch as _torch
        if not _torch.cuda.is_available():
            missing.append("a CUDA-capable GPU (torch.cuda.is_available() is "
                           "False; this build of torch may be the CPU-only one)")
    except Exception:                                   # pragma: no cover
        missing.append("torch with CUDA support")
    try:
        from cuquantum.bindings import custatevec        # type: ignore
    except Exception:
        try:
            from cuquantum import custatevec             # type: ignore
        except Exception:
            missing.append("the cuquantum-python package "
                           "(pip install cuquantum-python-cu12)")
            custatevec = None                            # type: ignore
    if missing:
        raise CuQuantumNotAvailable(
            "cuQuantum is not usable here: " + "; ".join(missing) + ". "
            "The CPU backends need none of this -- backend='statevector' "
            "gives the same numbers, more slowly.")
    return custatevec


class _Applier:
    """What the two layers agree on: make a state, apply a matrix, read it back.

    Deliberately small. Everything above it is shared and tested; everything
    below it is device-specific and is not.
    """

    def zero_state(self, n_qubits: int) -> Any:
        raise NotImplementedError

    def apply(self, state: Any, matrix: torch.Tensor,
              qubits: Sequence[int]) -> Any:
        raise NotImplementedError

    def to_host(self, state: Any) -> torch.Tensor:
        raise NotImplementedError


class _TorchApplier(_Applier):
    """The CPU mode: the same translation, applied with plain torch.

    It exists for two reasons that turn out to be one. A circuit written for
    `backend='cuquantum'` should run on a machine without a GPU, giving the
    same numbers more slowly, so that code does not have to be written twice.
    And because it runs the *same* translation, a test can compare it against
    `TorchBackend` and check every qubit-ordering decision on a machine with
    no GPU at all. If those two disagree, the translation is wrong -- on the
    device too.

    `device='cuda'` runs it on a GPU through torch alone, with no cuQuantum:
    useful for telling "the GPU is the problem" apart from "cuStateVec is the
    problem" when something does not agree.
    """

    def __init__(self, device: str = 'cpu') -> None:
        self.device = device

    def zero_state(self, n_qubits: int) -> torch.Tensor:
        state = torch.zeros(1 << n_qubits, dtype=torch.complex128,
                            device=self.device)
        state[0] = 1.0
        return state

    def apply(self, state: torch.Tensor, matrix: torch.Tensor,
              qubits: Sequence[int]) -> torch.Tensor:
        n_qubits = int(math.log2(state.shape[0]))
        matrix = matrix.to(dtype=torch.complex128, device=state.device)
        # Axis k of the reshaped state is qubit n-1-k, because qubit 0 is the
        # least-significant bit of the flat index.
        tensor = state.reshape((2, ) * n_qubits)
        axes = [n_qubits - 1 - q for q in qubits]
        return _apply_dense(tensor, matrix, axes).reshape(-1)

    def to_host(self, state: torch.Tensor) -> torch.Tensor:
        return state.detach().to('cpu')


def _apply_dense(tensor: torch.Tensor, matrix: torch.Tensor,
                 axes: Sequence[int]) -> torch.Tensor:
    """Contract `matrix` onto `axes` of `tensor`, leaving the axis order alone."""
    k = len(axes)
    reshaped = matrix.reshape((2, ) * (2 * k))
    moved = torch.movedim(tensor, list(axes), list(range(k)))
    out = torch.tensordot(reshaped, moved, dims=([*range(k, 2 * k)],
                                                 [*range(k)]))
    return torch.movedim(out, list(range(k)), list(axes))


class _CuStateVec(_Applier):
    """The cuStateVec calls, and nothing else.

    ⚠ Not exercised by the test suite: this repository's machine has no GPU,
    no CUDA and no cuQuantum, so every line below is written against the
    documented interface and has never been run. The layer above it is tested;
    this one needs a GPU and a person to say it worked.
    """

    def __init__(self) -> None:
        self._custatevec = _require_cuquantum()
        self._handle = self._custatevec.create()

    def __del__(self) -> None:                           # pragma: no cover
        try:
            self._custatevec.destroy(self._handle)
        except Exception:
            pass

    def zero_state(self, n_qubits: int) -> torch.Tensor:  # pragma: no cover
        state = torch.zeros(1 << n_qubits, dtype=torch.complex128, device='cuda')
        state[0] = 1.0
        return state

    def apply(self, state: torch.Tensor, matrix: torch.Tensor,
              qubits: Sequence[int]) -> torch.Tensor:      # pragma: no cover
        cusv = self._custatevec
        n_qubits = int(math.log2(state.shape[0]))
        host = matrix.to(dtype=torch.complex128, device='cpu').contiguous()
        # The whole operator goes down as one matrix, controls included. Handing
        # cuStateVec the controls separately would let it skip the untouched
        # half of the state and is the faster call -- but it is a second code
        # path, and a second code path that cannot be tested here is how the
        # two of them drift apart. Whoever measures this on a device can add
        # it, with a number showing it was worth it.
        cusv.apply_matrix(
            self._handle, state.data_ptr(), cusv.cudaDataType.CUDA_C_64F,
            n_qubits, host.numpy().ctypes.data,
            cusv.cudaDataType.CUDA_C_64F, cusv.MatrixLayout.ROW, 0,
            list(qubits), len(qubits), [], None, 0,
            cusv.ComputeType.COMPUTE_64F, 0, 0)
        return state

    def to_host(self, state: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return state.detach().to('cpu')


def _apply_all(gates: List[Operation], n_qubits: int, applier: _Applier,
               ) -> Tuple[Any, List[Tuple[int, Optional[str]]]]:
    """Run the gates through `applier`. Returns the state and the measurements.

    This is the whole translation: which matrix, on which qubits, in what
    order. It is device-independent on purpose -- a test runs it with the
    reference applier and requires the answer to match `TorchBackend`.
    """
    state = applier.zero_state(n_qubits)
    measured: List[Tuple[int, Optional[str]]] = []
    for gate in gates:
        name = gate.lowername
        if name in ('barrier', 'i'):
            continue
        if isinstance(gate, Measurement):
            for target in gate.target_iter(n_qubits):
                measured.append((int(target), getattr(gate, 'key', None)))
            continue
        matrix = None
        if hasattr(gate, 'matrix'):
            try:
                matrix = gate.matrix()
            except NotImplementedError:
                matrix = None
        if matrix is None:
            if isinstance(gate, IFallbackOperation):
                state, more = _apply_all(gate.fallback(n_qubits), n_qubits,
                                         _Prepared(applier, state))
                measured += more
                continue
            raise ValueError(
                f"the cuQuantum backend has no matrix for '{name}' and it "
                f"cannot be decomposed. Transpile it first.")
        for qubits in _qubits_for_matrix(gate, n_qubits):
            state = applier.apply(state, matrix, qubits)
    return state, measured


class _Prepared(_Applier):
    """An applier that starts from a state already in hand, for fallbacks."""

    def __init__(self, inner: _Applier, state: Any) -> None:
        self._inner, self._state = inner, state

    def zero_state(self, n_qubits: int) -> Any:
        return self._state

    def apply(self, state: Any, matrix: torch.Tensor,
              qubits: Sequence[int]) -> Any:
        return self._inner.apply(state, matrix, qubits)

    def to_host(self, state: Any) -> torch.Tensor:
        return self._inner.to_host(state)


def _qubits_for_matrix(gate: Operation, n_qubits: int) -> List[Tuple[int, ...]]:
    """Which qubits one operation's matrix acts on, per application.

    `gate.matrix()` is the full operator -- `CXGate` gives the 4x4, not X -- so
    a controlled gate needs no separate control argument. What it does need is
    the qubit order the matrix was written in, and that is *reversed* from the
    order the gate lists: measured against `TorchBackend`, applying a CX on
    (control, target) is wrong by 0.707 and on (target, control) is exact.
    Reading it the other way is the silent-mirror failure again, so it was
    settled by comparison rather than by reasoning about index conventions.
    """
    if hasattr(gate, 'control_target_iter'):
        try:
            pairs = list(gate.control_target_iter(n_qubits))
        except Exception:
            pairs = []
        if pairs:
            return [tuple(int(q) for q in pair)[::-1] for pair in pairs]
    targets = tuple(int(t) for t in gate.target_iter(n_qubits))
    width = int(math.log2(gate.matrix().shape[0]))
    if width == 1:
        return [(t, ) for t in targets]
    if len(targets) != width:
        raise ValueError(
            f"'{gate.lowername}' has a {gate.matrix().shape[0]}x"
            f"{gate.matrix().shape[0]} matrix but names {len(targets)} qubits.")
    return [targets[::-1]]


class CuQuantumBackend(Backend):
    """Statevector simulation through cuQuantum's cuStateVec.

    Accepts the arguments the CPU backends do -- ``shots``, ``seed``,
    ``bit_order``, ``returns='statevector'`` -- and answers in the same
    conventions, so a circuit does not have to be written differently for it.
    Sampling is done on the host from the returned amplitudes, which keeps the
    seeding identical to the CPU backends rather than introducing a second
    source of randomness that would have to be reconciled.

    `device` picks where the work happens:

    ``'auto'``
        cuStateVec if it is usable, and the CPU otherwise -- with a warning,
        because a silent fallback turns a GPU benchmark into a CPU one without
        saying so.
    ``'cpu'``
        The CPU, deliberately and quietly. Same numbers, more slowly, and it
        means a circuit written for this backend runs anywhere.
    ``'cuda'``
        cuStateVec, or an error naming what is missing. Nothing falls back.
    ``'torch-cuda'``
        A GPU through torch alone, without cuQuantum. Useful for telling "the
        GPU is the problem" apart from "cuStateVec is the problem".

    ⚠ The cuStateVec path has never been run: the machine this was written on
    has no GPU, no CUDA and no cuQuantum. Everything above it -- the
    translation, the conventions, the sampling -- is tested against
    `TorchBackend` and agrees to machine precision, and the CPU mode is that
    same tested path. The device calls are written against the documented
    interface and want a person with a GPU to confirm them.
    """

    def __init__(self, applier: Optional[_Applier] = None,
                 device: str = 'auto') -> None:
        if device not in ('auto', 'cuda', 'cpu', 'torch-cuda'):
            raise ValueError(
                f"device must be 'auto', 'cuda', 'cpu' or 'torch-cuda', "
                f"got {device!r}.")
        self._applier = applier
        self.device = device

    def _resolve(self, device: Optional[str] = None) -> _Applier:
        if self._applier is not None:
            return self._applier
        device = device or self.device
        if device == 'cpu':
            return _TorchApplier('cpu')
        if device == 'torch-cuda':
            return _TorchApplier('cuda')
        if device == 'cuda':
            return _CuStateVec()
        try:
            return _CuStateVec()
        except CuQuantumNotAvailable as reason:
            # Saying so matters more than the convenience of falling back.
            # Someone timing this needs to know they are timing a CPU, and a
            # silent fallback is exactly how a benchmark comes to mean
            # nothing.
            import warnings
            warnings.warn(
                f"running on the CPU: {reason} Pass device='cpu' to say so "
                f"deliberately and silence this, or device='cuda' to fail "
                f"instead of falling back.",
                RuntimeWarning, stacklevel=3)
            return _TorchApplier('cpu')

    def run(self, gates: List[Operation], n_qubits: int, *args: Any,
            **kwargs: Any) -> Any:
        returns = kwargs.get('returns')
        shots = kwargs.get('shots')
        bit_order = kwargs.get('bit_order', 'q0_last')
        seed = kwargs.get('seed')
        if returns not in (None, 'statevector', 'shots'):
            raise ValueError(
                f"returns={returns!r} is not supported by the cuQuantum "
                f"backend; it gives a statevector or shot counts.")

        applier = self._resolve(kwargs.get('device'))
        state, measured = _apply_all(list(gates), n_qubits, applier)
        amplitudes = applier.to_host(state)

        if shots is None and returns != 'shots':
            return amplitudes
        if shots is None:
            shots = 1
        return self._sample(amplitudes, measured, n_qubits, int(shots), seed,
                            bit_order)

    @staticmethod
    def _sample(amplitudes: torch.Tensor,
                measured: Sequence[Tuple[int, Optional[str]]], n_qubits: int,
                shots: int, seed: Optional[int], bit_order: str):
        """Counts, drawn on the host with blueqat's own conventions.

        Drawing here rather than with cuStateVec's sampler is a choice: the
        device sampler has its own generator, and two sources of randomness
        behind one `seed` argument is how a run stops being reproducible
        without anyone noticing.
        """
        from collections import Counter
        probabilities = (amplitudes.abs() ** 2).to(torch.float64)
        probabilities = probabilities / probabilities.sum()
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
        cumulative = torch.cumsum(probabilities, dim=0)
        cumulative[-1] = 1.0
        draws = torch.rand(shots, dtype=torch.float64, generator=generator)
        indices = torch.searchsorted(cumulative, draws)
        targets = sorted({q for q, _ in measured}) or list(range(n_qubits))
        counts: 'Counter[str]' = Counter()
        for index in indices.tolist():
            bits = ''.join('1' if (index >> q) & 1 else '0'
                           for q in reversed(range(n_qubits)))
            counts[''.join(bits[n_qubits - 1 - q] for q in reversed(targets))] += 1
        return apply_bit_order(counts, len(targets), bit_order)


register_backend('cuquantum', CuQuantumBackend, overwrite=True)
register_backend('custatevec', CuQuantumBackend, overwrite=True)
