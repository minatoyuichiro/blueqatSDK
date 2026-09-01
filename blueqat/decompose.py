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
"""

import math
from typing import Optional, Sequence, Tuple

import torch

from .circuit import Circuit

__all__ = ['two_qubit_kak', 'decompose_two_qubit']

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
