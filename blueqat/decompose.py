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
"""Turning a matrix into a circuit.

One-qubit matrices go straight in as ``mat1``. This module covers the
two-qubit case, where a general unitary factors as

    ``U = phase * (A1 (x) A2) exp(i(a XX + b YY + c ZZ)) (A3 (x) A4)``

-- the Cartan (KAK) decomposition. The interaction part is emitted as
``rxx``/``ryy``/``rzz``, and terms that vanish are left out, so a circuit built
from structured blocks costs less than the general bound.

That closed form costs six CX for a general unitary where three suffice.
:func:`synthesize_two_qubit` reaches three by fitting a three-CX circuit to the
target instead of solving for it, which is worth the iteration when the circuit
is bound for hardware and a CX budget is what decides whether the result
survives.
"""

import math
from typing import Optional, Sequence, Tuple

import torch

from .circuit import Circuit

__all__ = ['two_qubit_kak', 'decompose_two_qubit', 'synthesize_two_qubit',
           'decompose_unitary', 'cosine_sine', 'complete_to_unitary',
           'decompose_isometry']

_C = torch.complex128

#: Columns are the magic (Bell) basis. In it, SO(4) is exactly the group of
#: local two-qubit unitaries, which is what makes the decomposition work.
_MAGIC = torch.tensor([[1, 0, 0, 1j],
                       [0, 1j, 1, 0],
                       [0, 1j, -1, 0],
                       [1, 0, 0, -1j]], dtype=_C) / math.sqrt(2.0)

#: The magic-basis phases of ``exp(i(a XX + b YY + c ZZ))`` are linear in
#: (a, b, c); this is that map, measured rather than derived by hand.
_PHASE_MAP = torch.tensor([[1., -1., 1.],
                           [1., 1., -1.],
                           [-1., -1., -1.],
                           [-1., 1., 1.]], dtype=torch.float64)


def _tensor_factors(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Split ``kron(A, B)`` back into A and B.

    Also returns the second singular value of the rearranged matrix, which is
    zero exactly when the input really was a tensor product -- a cheap check
    that the caller has not been handed something entangling by mistake.
    """
    rearranged = matrix.reshape(2, 2, 2, 2).permute(0, 2, 1, 3).reshape(4, 4)
    u, s, vh = torch.linalg.svd(rearranged)
    scale = torch.sqrt(s[0].to(_C))
    return (u[:, 0] * scale).reshape(2, 2), (vh[0, :] * scale).reshape(2, 2), float(s[1])


def _real_orthogonal_diagonalizer(symmetric: torch.Tensor,
                                  attempts: int = 40) -> torch.Tensor:
    """A real orthogonal `P` with ``P.T @ symmetric @ P`` diagonal.

    `symmetric` is complex symmetric and unitary, so its real and imaginary
    parts are real symmetric and commute, and a single real orthogonal matrix
    diagonalizes both. Eigendecomposing one part alone fails whenever that part
    has a repeated eigenvalue, so a random combination of the two is used --
    generically non-degenerate, and checked rather than assumed.
    """
    real, imaginary = symmetric.real, symmetric.imag
    generator = torch.Generator().manual_seed(0)
    worst = float('inf')
    for _ in range(attempts):
        weights = torch.rand(2, generator=generator, dtype=torch.float64) * 2 - 1
        _, candidate = torch.linalg.eigh(weights[0] * real + weights[1] * imaginary)
        diagonalized = candidate.T.to(_C) @ symmetric @ candidate.to(_C)
        off = float((diagonalized
                     - torch.diag(torch.diagonal(diagonalized))).abs().max())
        worst = min(worst, off)
        if off < 1e-9:
            return candidate.to(_C)
    raise RuntimeError(
        f"Could not simultaneously diagonalize the magic-basis matrix "
        f"(best off-diagonal residual {worst:.2e}). Is the input unitary?")


def two_qubit_kak(matrix: torch.Tensor):
    """Factor a 4x4 unitary into local parts and a canonical interaction.

    Returns ``(left, (a, b, c), right, phase)`` with `left` and `right` local
    (each a tensor product) such that::

        matrix == phase * left @ expm(i(a XX + b YY + c ZZ)) @ right
    """
    matrix = torch.as_tensor(matrix, dtype=_C)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected a 4x4 matrix, got {tuple(matrix.shape)}.")
    if not torch.allclose(matrix @ matrix.conj().T,
                          torch.eye(4, dtype=_C), atol=1e-8):
        raise ValueError("matrix is not unitary.")

    phase = torch.linalg.det(matrix) ** 0.25
    special = matrix / phase                       # in SU(4)
    in_magic = _MAGIC.conj().T @ special @ _MAGIC
    symmetric = in_magic.T @ in_magic

    rotation = _real_orthogonal_diagonalizer(symmetric)
    if torch.linalg.det(rotation).real < 0:        # local parts live in SO(4)
        rotation = rotation.clone()
        rotation[:, 0] = -rotation[:, 0]
    diagonal = torch.sqrt(torch.diagonal(rotation.T @ symmetric @ rotation))
    left_magic = in_magic @ rotation @ torch.diag(1.0 / diagonal)
    if torch.linalg.det(left_magic).real < 0:
        diagonal = diagonal.clone()
        diagonal[0] = -diagonal[0]
        left_magic = in_magic @ rotation @ torch.diag(1.0 / diagonal)

    angles = torch.linalg.lstsq(
        _PHASE_MAP, torch.angle(diagonal).real.unsqueeze(1)).solution.squeeze()
    left = _MAGIC @ left_magic @ _MAGIC.conj().T
    right = _MAGIC @ rotation.T @ _MAGIC.conj().T
    return left, tuple(float(v) for v in angles), right, complex(phase)


def decompose_two_qubit(matrix: torch.Tensor,
                        targets: Sequence[int] = (0, 1),
                        n_qubits: Optional[int] = None,
                        atol: float = 1e-12) -> Circuit:
    """A circuit implementing a 4x4 unitary, exactly, up to global phase.

    `targets` names the two qubits as ``(low, high)``: the matrix is read in
    blueqat's convention, where ``targets[0]`` is the least significant bit of a
    basis-state index.

    Cost is three two-qubit rotations -- six CX once they are compiled -- and
    fewer when the interaction is degenerate, since a canonical angle within
    `atol` of zero contributes nothing and is dropped. A CZ, for instance, comes
    back as a single ``rzz``.
    """
    low, high = int(targets[0]), int(targets[1])
    if low == high:
        raise ValueError("the two targets must differ.")
    width = max(low, high) + 1 if n_qubits is None else int(n_qubits)
    if width <= max(low, high):
        raise ValueError(f"n_qubits={width} cannot hold targets {targets}.")

    left, (a, b, c), right, _phase = two_qubit_kak(matrix)
    first_high, first_low, residual_r = _tensor_factors(right)
    last_high, last_low, residual_l = _tensor_factors(left)
    worst = max(residual_r, residual_l)
    if worst > 1e-8:
        raise RuntimeError(
            f"the local factors did not come out separable (residual {worst:.2e}); "
            f"this is a bug in the decomposition, not in the input.")

    circuit = Circuit(width)
    circuit.mat1(first_high)[high]
    circuit.mat1(first_low)[low]
    # rxx(t) is exp(-i t/2 XX), so exp(i a XX) is rxx(-2a).
    for angle, name in ((a, 'rxx'), (b, 'ryy'), (c, 'rzz')):
        if abs(angle) > atol:
            getattr(circuit, name)(-2.0 * angle)[low, high]
    circuit.mat1(last_high)[high]
    circuit.mat1(last_low)[low]
    return circuit


def _single_qubit_from_euler(params: torch.Tensor) -> torch.Tensor:
    """``u(theta, phi, lam)`` as a 2x2 matrix, differentiably."""
    theta, phi, lam = params[0], params[1], params[2]
    cos = torch.cos(theta / 2).to(_C)
    sin = torch.sin(theta / 2).to(_C)
    return torch.stack([
        torch.stack([cos, -torch.exp(1j * lam.to(_C)) * sin]),
        torch.stack([torch.exp(1j * phi.to(_C)) * sin,
                     torch.exp(1j * (phi + lam).to(_C)) * cos])])


_CX_HIGH_TO_LOW = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0],
                                [0, 0, 0, 1], [0, 0, 1, 0]], dtype=_C)
_CX_LOW_TO_HIGH = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 1],
                                [0, 0, 1, 0], [0, 1, 0, 0]], dtype=_C)


def _three_cx_ansatz(params: torch.Tensor, n_cx: int) -> torch.Tensor:
    """Alternating CX and layers of arbitrary single-qubit gates."""
    layers = params.reshape(-1, 3)
    entanglers = [_CX_HIGH_TO_LOW, _CX_LOW_TO_HIGH, _CX_HIGH_TO_LOW]
    unitary = torch.kron(_single_qubit_from_euler(layers[0]),
                         _single_qubit_from_euler(layers[1]))
    for step in range(n_cx):
        unitary = entanglers[step] @ unitary
        unitary = torch.kron(_single_qubit_from_euler(layers[2 * step + 2]),
                             _single_qubit_from_euler(layers[2 * step + 3])) @ unitary
    return unitary


def synthesize_two_qubit(matrix: torch.Tensor,
                         targets: Sequence[int] = (0, 1),
                         n_qubits: Optional[int] = None,
                         n_cx: int = 3,
                         restarts: int = 6,
                         tol: float = 1e-10) -> Circuit:
    """A circuit for a 4x4 unitary using at most `n_cx` CX gates.

    Three CX suffice for any two-qubit unitary, against the six that the closed
    form in :func:`decompose_two_qubit` emits. Getting there in closed form
    means folding the canonical angles into the Weyl chamber with matching local
    corrections; this fits the circuit to the target instead, by gradient
    descent on the single-qubit layers between the CX gates.

    The result is checked, not assumed: the fit must reach an infidelity below
    `tol` or this raises. Use :func:`decompose_two_qubit` when an exact closed
    form matters more than the gate count.
    """
    matrix = torch.as_tensor(matrix, dtype=_C)
    if matrix.shape != (4, 4):
        raise ValueError(f"expected a 4x4 matrix, got {tuple(matrix.shape)}.")
    if not 0 <= n_cx <= 3:
        raise ValueError(f"n_cx must be between 0 and 3, got {n_cx}.")
    low, high = int(targets[0]), int(targets[1])
    if low == high:
        raise ValueError("the two targets must differ.")
    width = max(low, high) + 1 if n_qubits is None else int(n_qubits)
    if width <= max(low, high):
        raise ValueError(f"n_qubits={width} cannot hold targets {targets}.")

    generator = torch.Generator().manual_seed(0)
    size = (n_cx + 1) * 2 * 3
    best_error, best_params = float('inf'), None
    for _ in range(restarts):
        params = (torch.rand(size, generator=generator, dtype=torch.float64)
                  * 2 * math.pi).requires_grad_(True)

        def infidelity() -> torch.Tensor:
            overlap = torch.trace(matrix.conj().T @ _three_cx_ansatz(params, n_cx))
            return 1.0 - torch.abs(overlap) / 4.0

        adam = torch.optim.Adam([params], lr=0.15)
        for _ in range(400):
            adam.zero_grad()
            loss = infidelity()
            loss.backward()
            adam.step()
            if float(loss.detach()) < tol * 1e-2:
                break
        polish = torch.optim.LBFGS([params], max_iter=200, tolerance_grad=1e-14,
                                   tolerance_change=1e-16)

        def closure() -> torch.Tensor:
            polish.zero_grad()
            loss = infidelity()
            loss.backward()
            return loss

        polish.step(closure)
        with torch.no_grad():
            error = float(infidelity())
        if error < best_error:
            best_error, best_params = error, params.detach().clone()
        if best_error < tol * 1e-2:
            break

    if best_error > tol:
        raise RuntimeError(
            f"could not fit a {n_cx}-CX circuit to this unitary (infidelity "
            f"{best_error:.2e} against a tolerance of {tol:.0e}). "
            f"decompose_two_qubit solves it exactly at six CX.")

    layers = best_params.reshape(-1, 3)
    circuit = Circuit(width)
    entanglers = [(high, low), (low, high), (high, low)]
    circuit.mat1(_single_qubit_from_euler(layers[0]))[high]
    circuit.mat1(_single_qubit_from_euler(layers[1]))[low]
    for step in range(n_cx):
        control, target = entanglers[step]
        circuit.cx[control, target]
        circuit.mat1(_single_qubit_from_euler(layers[2 * step + 2]))[high]
        circuit.mat1(_single_qubit_from_euler(layers[2 * step + 3]))[low]
    return circuit


# ---------------------------------------------------------------------------
# Any number of qubits: the Quantum Shannon decomposition.
# ---------------------------------------------------------------------------

def cosine_sine(matrix: torch.Tensor):
    """The cosine-sine decomposition of a unitary of even size.

    ``matrix == blockdiag(L0, L1) @ [[C, -S], [S, C]] @ blockdiag(R0, R1)^H``
    with `C` and `S` diagonal and non-negative. Returns
    ``(L0, L1, cosines, sines, R0, R1)``.

    SciPy's ``linalg.cossin`` is used when it is installed, because it handles
    the degenerate cases -- repeated cosines, and columns where the sine
    vanishes -- that a plain SVD-based construction does not. Those are not
    exotic: a Toffoli gate and the unitary completion of an isometry both hit
    them. Without SciPy the fallback below covers matrices whose cosines are
    distinct, and says so rather than returning factors that are quietly not
    unitary.
    """
    matrix = torch.as_tensor(matrix, dtype=_C)
    size = matrix.shape[0]
    if size % 2:
        raise ValueError(f"matrix size must be even, got {size}.")
    half = size // 2

    try:
        from scipy.linalg import cossin as _scipy_cossin
    except ImportError:
        pass
    else:
        (left_top, left_bottom), angles, (right_top_h, right_bottom_h) = _scipy_cossin(
            matrix.numpy(), p=half, q=half, separate=True)
        to_tensor = lambda a: torch.as_tensor(a, dtype=_C)
        cosines = torch.as_tensor(angles, dtype=torch.float64).cos()
        sines = torch.as_tensor(angles, dtype=torch.float64).sin()
        return (to_tensor(left_top), to_tensor(left_bottom), cosines, sines,
                to_tensor(right_top_h).conj().T, to_tensor(right_bottom_h).conj().T)

    upper_left, upper_right = matrix[:half, :half], matrix[:half, half:]
    lower_left, lower_right = matrix[half:, :half], matrix[half:, half:]

    left_top, cosines, right_top_h = torch.linalg.svd(upper_left)
    order = torch.argsort(cosines)          # cosines ascending, angles descending
    cosines = cosines[order]
    left_top = left_top[:, order]
    right_top = right_top_h.conj().T[:, order]
    sines = torch.sqrt(torch.clamp(1 - cosines ** 2, min=0.0))

    # The second left factor follows from lower_left, except where the sine
    # vanishes and the column carries no information; those are filled with any
    # vectors completing the basis.
    rotated = lower_left @ right_top
    left_bottom = torch.zeros_like(left_top)
    determined = [k for k in range(half) if sines[k] > 1e-9]
    undetermined = [k for k in range(half) if sines[k] <= 1e-9]
    for k in determined:
        left_bottom[:, k] = rotated[:, k] / sines[k].to(_C)
    if undetermined:
        known = (left_bottom[:, determined] if determined
                 else torch.zeros((half, 0), dtype=_C))
        filler = torch.linalg.qr(
            torch.cat([known, torch.eye(half, dtype=_C)], dim=1))[0]
        for offset, k in enumerate(undetermined):
            left_bottom[:, k] = filler[:, len(determined) + offset]

    right_bottom = torch.zeros_like(right_top)
    for k in range(half):
        if cosines[k] > 1e-9:
            right_bottom[:, k] = (lower_right.conj().T @ left_bottom[:, k]) / cosines[k].to(_C)
        else:
            right_bottom[:, k] = -(upper_right.conj().T @ left_top[:, k]) / sines[k].to(_C)

    # This construction is only valid when the cosines are distinct; where they
    # repeat it silently produces factors that are not unitary, so check instead
    # of hoping. Installing SciPy takes the exact path above.
    identity = torch.eye(half, dtype=_C)
    for name, factor in (('left', left_bottom), ('right', right_bottom)):
        if not torch.allclose(factor.conj().T @ factor, identity, atol=1e-7):
            raise RuntimeError(
                f"the cosine-sine decomposition's {name} factor came out non-unitary. "
                f"This matrix has repeated cosines, which the built-in construction "
                f"cannot separate; install SciPy (pip install scipy) for the exact "
                f"decomposition.")
    return left_top, left_bottom, cosines, sines, right_top, right_bottom


def _demultiplex(first: torch.Tensor, second: torch.Tensor):
    """``blockdiag(first, second) == (I (x) v) blockdiag(d, d^H) (I (x) w)``.

    From ``first @ second^H = v d^2 v^H``. The eigenvectors of a unitary are not
    orthonormal by default where eigenvalues repeat, so they are orthonormalized
    and the diagonal is then read back off rather than taken from the solver.
    """
    product = first @ second.conj().T
    vectors = _unitary_eigenbasis(product)
    squared = torch.diagonal(vectors.conj().T @ product @ vectors)
    diagonal = torch.sqrt(squared)
    return vectors, diagonal, torch.diag(diagonal) @ vectors.conj().T @ second


def _unitary_eigenbasis(product: torch.Tensor, attempts: int = 40) -> torch.Tensor:
    """An orthonormal eigenbasis of a unitary matrix.

    ``torch.linalg.eig`` returns eigenvectors that are not orthonormal wherever
    eigenvalues repeat, and orthonormalizing them afterwards mixes eigenspaces
    rather than fixing them -- which shows up downstream as a factor that is no
    longer unitary. A unitary's Hermitian and anti-Hermitian parts are Hermitian
    and commute, so `eigh` on a random real combination of the two gives a basis
    diagonalizing both, and therefore the unitary itself.
    """
    hermitian = (product + product.conj().T) / 2
    anti = (product - product.conj().T) / 2j
    generator = torch.Generator().manual_seed(0)
    worst = float('inf')
    for _ in range(attempts):
        weights = torch.rand(2, generator=generator, dtype=torch.float64) * 2 - 1
        _, vectors = torch.linalg.eigh(weights[0] * hermitian + weights[1] * anti)
        diagonalized = vectors.conj().T @ product @ vectors
        off = float((diagonalized
                     - torch.diag(torch.diagonal(diagonalized))).abs().max())
        worst = min(worst, off)
        if off < 1e-9:
            return vectors
    raise RuntimeError(
        f"could not find an orthonormal eigenbasis (residual {worst:.2e})")


def _gray(index: int) -> int:
    return index ^ (index >> 1)


def _uniformly_controlled_angles(angles: Sequence[float]) -> torch.Tensor:
    """Rotation angles for the alternating rotation/CX chain that realizes a
    rotation whose angle depends on the control register's basis state."""
    count = len(angles)
    controls = count.bit_length() - 1
    transform = torch.empty((count, count), dtype=torch.float64)
    for i in range(count):
        for j in range(count):
            transform[i, j] = (-1.0) ** bin(i & _gray(j)).count('1')
    del controls
    return torch.linalg.solve(transform, torch.as_tensor(angles, dtype=torch.float64))


def _uniformly_controlled(circuit: Circuit, kind: str, angles: Sequence[float],
                          target: int, controls: Sequence[int]) -> None:
    resolved = _uniformly_controlled_angles(angles)
    count = len(angles)
    for j in range(count):
        getattr(circuit, kind)(float(resolved[j]))[target]
        if count > 1:
            flipped = _gray(j) ^ _gray((j + 1) % count)
            circuit.cx[controls[flipped.bit_length() - 1], target]


def decompose_unitary(matrix: torch.Tensor,
                      targets: Optional[Sequence[int]] = None,
                      n_qubits: Optional[int] = None) -> Circuit:
    """A circuit for an arbitrary ``2**n x 2**n`` unitary, exactly.

    The Quantum Shannon decomposition: split the matrix on its top qubit with a
    cosine-sine decomposition, turn the two block-diagonal halves into a
    controlled rotation plus smaller unitaries, and recurse. The recursion stops
    at two qubits, where :func:`decompose_two_qubit` solves it in closed form.

    Cost grows as ``4**n``; this is a correct construction, not an optimized
    one. For a single two-qubit block prefer :func:`synthesize_two_qubit`,
    which reaches the optimal three CX.
    """
    matrix = torch.as_tensor(matrix, dtype=_C)
    size = matrix.shape[0]
    n = size.bit_length() - 1
    if matrix.shape != (size, size) or (1 << n) != size:
        raise ValueError(f"expected a 2**n x 2**n matrix, got {tuple(matrix.shape)}.")
    if not torch.allclose(matrix @ matrix.conj().T, torch.eye(size, dtype=_C), atol=1e-8):
        raise ValueError("matrix is not unitary.")
    if targets is None:
        targets = list(range(n))
    if len(targets) != n:
        raise ValueError(f"expected {n} targets for a {size}x{size} matrix.")
    width = max(targets) + 1 if n_qubits is None else int(n_qubits)

    circuit = Circuit(width)
    circuit.ops.extend(_shannon(matrix, list(targets), width))
    return circuit


def _shannon(matrix: torch.Tensor, targets: Sequence[int], width: int) -> list:
    n = len(targets)
    if n == 1:
        circuit = Circuit(width)
        circuit.mat1(matrix)[targets[0]]
        return circuit.ops
    if n == 2:
        return decompose_two_qubit(matrix, targets=(targets[0], targets[1]),
                                   n_qubits=width).ops

    top, rest = targets[-1], list(targets[:-1])
    left_top, left_bottom, cosines, sines, right_top, right_bottom = cosine_sine(matrix)
    ops: list = []

    def multiplexed(first, second):
        vectors, diagonal, other = _demultiplex(first, second)
        block: list = []
        block += _shannon(other, rest, width)
        rotations = Circuit(width)
        _uniformly_controlled(rotations, 'rz',
                              [-2.0 * float(torch.angle(value)) for value in diagonal],
                              top, rest)
        block += rotations.ops
        block += _shannon(vectors, rest, width)
        return block

    ops += multiplexed(right_top.conj().T, right_bottom.conj().T)
    middle = Circuit(width)
    _uniformly_controlled(middle, 'ry',
                          [2.0 * math.atan2(float(sines[k]), float(cosines[k]))
                           for k in range(len(cosines))], top, rest)
    ops += middle.ops
    ops += multiplexed(left_top, left_bottom)
    return ops


def complete_to_unitary(isometry: torch.Tensor) -> torch.Tensor:
    """Extend a ``2**n x 2**k`` isometry to a ``2**n x 2**n`` unitary.

    The extra columns are any orthonormal basis of the orthogonal complement of
    the isometry's column space; which one is chosen does not matter, because
    the circuit only ever sees them applied to inputs that are zero there.
    """
    isometry = torch.as_tensor(isometry, dtype=_C)
    rows, cols = isometry.shape
    if not torch.allclose(isometry.conj().T @ isometry,
                          torch.eye(cols, dtype=_C), atol=1e-8):
        raise ValueError("the columns are not orthonormal; this is not an isometry.")
    if cols == rows:
        return isometry
    projector = torch.eye(rows, dtype=_C) - isometry @ isometry.conj().T
    vectors, values, _ = torch.linalg.svd(projector)
    complement = vectors[:, :rows - cols]
    del values
    return torch.cat([isometry, complement], dim=1)


def decompose_isometry(isometry: torch.Tensor,
                       targets: Optional[Sequence[int]] = None,
                       n_qubits: Optional[int] = None) -> Circuit:
    """A circuit applying a ``2**n x 2**k`` isometry.

    The circuit acts on `n` qubits and reproduces the isometry **when the
    qubits above the input register start in** ``|0>`` -- that is, on `k` input
    qubits padded with ``n - k`` fresh ones. This is the shape that turns up
    when a matrix product state is written as a sequential circuit, where each
    site's tensor is an isometry from the bond to the bond plus the new site.

    Built by completing the isometry to a unitary and decomposing that, so the
    cost is a full unitary's; the columns the padding never reaches are free to
    be anything, and no attempt is made to exploit that.
    """
    isometry = torch.as_tensor(isometry, dtype=_C)
    rows, cols = isometry.shape
    n = rows.bit_length() - 1
    if (1 << n) != rows or (cols & (cols - 1)):
        raise ValueError(
            f"expected a 2**n x 2**k isometry, got {tuple(isometry.shape)}.")
    if cols > rows:
        raise ValueError("an isometry cannot have more columns than rows.")
    return decompose_unitary(complete_to_unitary(isometry),
                             targets=targets, n_qubits=n_qubits)
