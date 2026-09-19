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
"""Differentiable synthesis of logical EO gates as short exchange-pulse
sequences, using PyTorch autograd (the whole pipeline -- pulse areas ->
exchange matrices -> logical block -> fidelity -- is differentiable).

This is what allows going beyond the fixed analytic gate tables: any target
SU(2) can be compiled into a few constant-amplitude pulses."""

import cmath
import math
from typing import List, Optional, Sequence, Tuple

import torch

from .encoding import codeword_basis

Pulse = Tuple[Tuple[int, int], float]

_TWO_PI = 2.0 * math.pi


def _exchange_matrix(theta: torch.Tensor) -> torch.Tensor:
    """Differentiable 4x4 exchange unitary (same convention as ExchangeGate)."""
    one = torch.ones((), dtype=torch.complex128)
    zero = torch.zeros((), dtype=torch.complex128)
    e = torch.exp(1j * theta.to(torch.complex128))
    a = (one + e) * 0.5
    b = (one - e) * 0.5
    return torch.stack([
        one, zero, zero, zero,
        zero, a, b, zero,
        zero, b, a, zero,
        zero, zero, zero, one
    ]).reshape(4, 4)


def _pulse_unitary_3spin(pair: Tuple[int, int], theta: torch.Tensor) -> torch.Tensor:
    """8x8 unitary of one exchange pulse on a 3-spin triple."""
    e = _exchange_matrix(theta)
    eye2 = torch.eye(2, dtype=torch.complex128)
    if pair == (0, 1):
        return torch.kron(eye2, e)
    if pair == (1, 2):
        return torch.kron(e, eye2)
    raise ValueError('pair must be (0, 1) or (1, 2).')


def synthesize_1q(target: torch.Tensor,
                  n_pulses: int = 4,
                  n_restarts: int = 8,
                  max_iter: int = 400,
                  fidelity_goal: float = 1.0 - 1e-9,
                  seed: Optional[int] = 0,
                  offset: int = 0) -> List[Pulse]:
    """Synthesize a logical 1-qubit gate as `n_pulses` exchange pulses
    alternating on pairs (0,1) and (1,2) of one triple.

    Returns the pulse sequence in application order (compatible with
    `sequences.sequence_to_circuit`). Raises RuntimeError if no restart
    reaches `fidelity_goal` -- some targets need more pulses (4 suffices
    for generic SU(2) with these two 120-degree-tilted rotation axes).
    """
    target = torch.as_tensor(target, dtype=torch.complex128)
    if target.shape != (2, 2):
        raise ValueError('target must be a 2x2 unitary.')
    basis = codeword_basis('+')
    pairs = [(0, 1) if k % 2 == 0 else (1, 2) for k in range(n_pulses)]

    if seed is not None:
        torch.manual_seed(seed)

    best: Optional[Tuple[float, torch.Tensor]] = None
    for _ in range(n_restarts):
        thetas = torch.rand(n_pulses, dtype=torch.float64) * _TWO_PI
        thetas.requires_grad_(True)
        opt = torch.optim.Adam([thetas], lr=0.1)
        for _ in range(max_iter):
            opt.zero_grad()
            u = torch.eye(8, dtype=torch.complex128)
            for pair, th in zip(pairs, thetas):
                u = _pulse_unitary_3spin(pair, th) @ u
            block = basis.conj().T @ u @ basis
            tr = torch.trace(block.conj().T @ target)
            loss = 1.0 - (tr.abs() ** 2) / 4.0
            loss.backward()
            opt.step()
            if loss.item() < 1.0 - fidelity_goal:
                break
        fid = 1.0 - loss.item()
        if best is None or fid > best[0]:
            best = (fid, thetas.detach().clone())
        if fid >= fidelity_goal:
            break

    assert best is not None
    fid, thetas = best
    if fid < fidelity_goal:
        raise RuntimeError(
            f'synthesis reached fidelity {fid:.12f} < goal {fidelity_goal}; '
            'try more pulses (n_pulses) or more restarts.')
    return [((offset + pair[0], offset + pair[1]), float(th) % _TWO_PI)
            for pair, th in zip(pairs, thetas)]


def _pulse_unitary(n_spins: int, pair: Tuple[int, int],
                   theta: torch.Tensor) -> torch.Tensor:
    """Differentiable 2**n x 2**n unitary of one exchange pulse on any pair:
    U = I + (e^{i theta} - 1)(I - SWAP_ij)/2, with SWAP_ij the (constant)
    bit-swap permutation, so autograd only flows through e^{i theta}."""
    i, j = pair
    dim = 1 << n_spins
    idx = torch.arange(dim)
    diff = ((idx >> i) & 1) ^ ((idx >> j) & 1)
    swapped = idx ^ ((diff << i) | (diff << j))
    swap = torch.zeros(dim, dim, dtype=torch.complex128)
    swap[idx, swapped] = 1.0
    eye = torch.eye(dim, dtype=torch.complex128)
    e = torch.exp(1j * theta.to(torch.complex128))
    return eye + (e - 1.0) * (eye - swap) * 0.5


def _sequence_unitary(n_spins: int, pairs: Sequence[Tuple[int, int]],
                      thetas: torch.Tensor) -> torch.Tensor:
    u = torch.eye(2 ** n_spins, dtype=torch.complex128)
    for pair, th in zip(pairs, thetas):
        u = _pulse_unitary(n_spins, pair, th) @ u
    return u


def synthesize_2q(target: torch.Tensor,
                  pairs: Sequence[Tuple[int, int]],
                  initial_thetas: Optional[Sequence[float]] = None,
                  n_restarts: int = 4,
                  max_iter: int = 1000,
                  fidelity_goal: float = 1.0 - 1e-8,
                  seed: Optional[int] = 0) -> List[Pulse]:
    """Synthesize an encoded 2-logical-qubit gate (logical qubit 0 on spins
    0-2, logical qubit 1 on spins 3-5) as exchange pulses on the given pair
    pattern.

    The loss demands a *gauge-independent, gauge-preserving* implementation:
    the logical block must equal `target` with one common phase in all four
    total-Sz sectors (leakage automatically suppresses the fidelity, so it
    needs no separate penalty). Note that some natural constructions are
    gauge-*permuting* instead -- e.g. the 3-pulse physical triple swap
    realizes an encoded SWAP but exchanges the two gauge states with it --
    and such gates cannot (and need not) be found by this loss.

    Pass `initial_thetas` to refine a known sequence -- e.g. to re-calibrate
    the Fong-Wandzura angles after hardware perturbations -- instead of
    starting from random pulses; from-scratch synthesis of long 2-qubit
    sequences is a hard non-convex problem and may need many restarts.
    """
    from .encoding import two_qubit_codeword_basis
    target = torch.as_tensor(target, dtype=torch.complex128)
    if target.shape != (4, 4):
        raise ValueError('target must be a 4x4 unitary.')
    bases = [two_qubit_codeword_basis(m1, m2)
             for m1 in ('+', '-') for m2 in ('+', '-')]
    n_pulses = len(pairs)

    if seed is not None:
        torch.manual_seed(seed)

    def loss_of(thetas: torch.Tensor) -> torch.Tensor:
        u = _sequence_unitary(6, pairs, thetas)
        # Sum the per-sector trace overlaps BEFORE taking |.|: this forces a
        # single common phase across sectors (true gauge independence).
        tr_sum = torch.zeros((), dtype=torch.complex128)
        for basis in bases:
            block = basis.conj().T @ u @ basis
            tr_sum = tr_sum + torch.trace(block.conj().T @ target)
        return 1.0 - (tr_sum.abs() ** 2) / (4 * 4) ** 2

    best: Optional[Tuple[float, torch.Tensor]] = None
    for restart in range(n_restarts):
        if initial_thetas is not None and restart == 0:
            thetas = torch.tensor(list(initial_thetas), dtype=torch.float64)
        else:
            thetas = torch.rand(n_pulses, dtype=torch.float64) * _TWO_PI
        thetas.requires_grad_(True)

        # From a poor (random) start, Adam explores the non-convex landscape;
        # from a good start (refinement), it would only wander, so skip it.
        if loss_of(thetas.detach()).item() > 1e-2:
            opt = torch.optim.Adam([thetas], lr=0.05)
            for _ in range(max_iter):
                opt.zero_grad()
                loss = loss_of(thetas)
                loss.backward()
                opt.step()
                if loss.item() < 1.0 - fidelity_goal:
                    break

        # L-BFGS with a strong-Wolfe line search converges the final digits
        # (crucial for the calibration-refinement use case).
        polish = torch.optim.LBFGS([thetas], max_iter=200,
                                   tolerance_grad=1e-15, tolerance_change=0,
                                   line_search_fn='strong_wolfe')

        def _closure():
            polish.zero_grad()
            l = loss_of(thetas)
            l.backward()
            return l

        polish.step(_closure)
        fid = 1.0 - loss_of(thetas.detach()).item()
        if best is None or fid > best[0]:
            best = (fid, thetas.detach().clone())
        if fid >= fidelity_goal:
            break

    assert best is not None
    fid, thetas = best
    if fid < fidelity_goal:
        raise RuntimeError(
            f'2q synthesis reached fidelity {fid:.12f} < goal {fidelity_goal}; '
            'try a different pair pattern, more pulses, or more restarts.')
    return [(tuple(pair), float(th) % _TWO_PI)
            for pair, th in zip(pairs, thetas)]


def quantize_sequence(sequence: Sequence[Pulse], step: float) -> List[Pulse]:
    """Snap every pulse area to the nearest multiple of `step` and drop
    pulses that round to zero -- the operational constraint of constant-
    amplitude hardware whose pulse durations come in discrete clock ticks.

    Check the result's fidelity yourself (e.g. via `encoding.logical_action`);
    a coarse step degrades the gate."""
    if step <= 0:
        raise ValueError('step must be positive.')
    out: List[Pulse] = []
    for pair, theta in sequence:
        q = round((theta % _TWO_PI) / step) * step
        if abs(q) < step / 2 or abs(q - _TWO_PI) < step / 2:
            continue
        out.append((pair, q))
    return out

#: The two exchange pairs of a triple act on the logical qubit as rotations
#: about axes 120 degrees apart -- pair (0,1) about -z, pair (1,2) about
#: ``(sqrt(3)/2, 0, 1/2)`` -- and by exactly the pulse area, measured. That
#: fixes what three pulses can reach: the off-diagonal entry of a product
#: ``Ra Rb Ra`` has modulus ``(sqrt(3)/2) |sin(beta/2)|``, and the outer two
#: pulses are diagonal, so they cannot change it.
THREE_PULSE_REACH = math.sqrt(3) / 2


def _logical(pair: Tuple[int, int], theta: float) -> torch.Tensor:
    """The 2x2 logical action of one exchange pulse."""
    unitary = _pulse_unitary_3spin(pair, torch.as_tensor(float(theta),
                                                         dtype=torch.float64))
    basis = codeword_basis('+')
    return basis.conj().T @ unitary @ basis


def _solve_three(target: torch.Tensor, offset: int = 0
                 ) -> Optional[List[Pulse]]:
    """Three pulses for `target`, in closed form, or None if out of reach.

    Pair (0,1) acts diagonally -- measured as ``diag(exp(i*theta), 1)``, with a
    global phase and the full pulse area rather than half of it -- so in
    ``Ra(a) Rb(b) Ra(g)`` the outer pulses only move phases about. The modulus
    of the off-diagonal entry therefore comes from the middle pulse alone and
    fixes it: ``|U01| = (sqrt(3)/2) |sin(beta/2)|``. The outer angles then come
    from two entry phases. No optimizer, no restarts, and the reachability
    condition falls out of the same equation rather than being discovered by a
    fit failing to converge.
    """
    target = torch.as_tensor(target, dtype=torch.complex128)
    off = abs(complex(target[0, 1]))
    if off > THREE_PULSE_REACH + 1e-12:
        return None
    base = 2.0 * math.asin(min(1.0, off / THREE_PULSE_REACH))
    low, high = offset, offset + 1
    for beta in (base, -base, 2 * math.pi - base, base - 2 * math.pi):
        middle = _logical((1, 2), beta)
        for i, j in ((1, 1), (0, 0)):
            if (abs(complex(target[i, j])) < 1e-9
                    or abs(complex(middle[i, j])) < 1e-9):
                continue
            phase = complex(middle[i, j]) / complex(target[i, j])
            if abs(complex(middle[0, 1])) < 1e-12:
                # A vanishing middle pulse leaves a diagonal product: one pulse.
                alpha = (cmath.phase(phase * complex(target[0, 0]))
                         - cmath.phase(complex(middle[0, 0])))
                gamma = 0.0
            else:
                alpha = (cmath.phase(phase * complex(target[0, 1]))
                         - cmath.phase(complex(middle[0, 1])))
                gamma = (cmath.phase(phase * complex(target[1, 0]))
                         - cmath.phase(complex(middle[1, 0])))
            product = (_logical((0, 1), alpha) @ _logical((1, 2), beta)
                       @ _logical((0, 1), gamma))
            if abs(complex(torch.trace(product.conj().T @ target))) / 2 > 1 - 1e-9:
                # Matrix order is the reverse of the order pulses are applied.
                return [((low, high), _wrap(gamma)),
                        ((high, high + 1), _wrap(beta)),
                        ((low, high), _wrap(alpha))]
    return None


def _wrap(theta: float) -> float:
    """Into [0, 2*pi), which is the period of the exchange unitary."""
    return float(theta % (2.0 * math.pi))


def decompose_1q(target: torch.Tensor, offset: int = 0,
                 samples: int = 720) -> List[Pulse]:
    """A logical 1-qubit gate as three or four exchange pulses, in closed form.

    Exact, and far shorter than composing the analytic tables: measured against
    them, ``rx`` goes from seven pulses to three or four and ``ry`` from nine.
    On exchange-only hardware a pulse is a gate, so that is the error budget.

    Three suffice when ``|target[0,1]| <= sqrt(3)/2``, which for ``rx(theta)``
    means ``theta <= 2*pi/3``. Beyond it a fourth pulse is applied first to
    bring the target inside: measured over random unitaries outside the reach,
    the largest residual after the best fourth pulse is 0.829 against the 0.866
    bound, so four always suffice. `samples` is how finely that fourth angle is
    searched before being refined; it is a one-dimensional minimum, not a fit
    over the whole sequence.
    """
    target = torch.as_tensor(target, dtype=torch.complex128)
    if target.shape != (2, 2):
        raise ValueError('target must be a 2x2 unitary.')
    direct = _solve_three(target, offset)
    if direct is not None:
        return direct

    best = (float('inf'), 0.0)
    for k in range(samples):
        delta = 2.0 * math.pi * k / samples
        moved = target @ _logical((1, 2), -delta)
        value = abs(complex(moved[0, 1]))
        if value < best[0]:
            best = (value, delta)
    # Refine around the coarse minimum; the function is smooth in delta.
    step = 2.0 * math.pi / samples
    delta = best[1]
    for _ in range(40):
        step /= 2.0
        for candidate in (delta - step, delta + step):
            value = abs(complex((target @ _logical((1, 2), -candidate))[0, 1]))
            if value < best[0]:
                best, delta = (value, candidate), candidate
    rest = _solve_three(target @ _logical((1, 2), -delta), offset)
    if rest is None:
        raise RuntimeError(
            f"no four-pulse sequence found: the best fourth angle leaves "
            f"|target[0,1]| = {best[0]:.4f}, above the {THREE_PULSE_REACH:.4f} "
            f"that three pulses reach. This should not happen and is a bug.")
    return [((offset + 1, offset + 2), _wrap(delta))] + rest
