"""Shor's algorithm (order finding for N = 15, a = 2) built from named blocks.

Shor's algorithm is naturally a nest of subroutines:

    order-finding
    ├─ superposition        (H on the counting register)
    ├─ c-U^(2^k)            (controlled modular multiplications)
    └─ IQFT                 (inverse QFT reads out the phase)

Named blocks (`with c.block(...)`) keep exactly that structure in the circuit
object -- `print(c.tree())` shows the nesting -- while every backend still
executes the underlying gates transparently.

Here U|x> = |2x mod 15>, which on 4 bits is just a cyclic left-shift of the
bit register, so the controlled versions are short cswap chains. The measured
phase s/r with r = 4 gives the factors gcd(2^(r/2) +- 1, 15) = 3 and 5.
"""
import math
from math import gcd

import torch

from blueqat import Circuit

N = 15
A = 2
N_WORK = 4          # work register: qubits 0..3, holds x
N_COUNT = 3         # counting register: qubits 4..6
COUNT = list(range(N_WORK, N_WORK + N_COUNT))


def qft_circuit(n: int) -> Circuit:
    """QFT on qubits 0..n-1 (kept as a library circuit; placed via append_block)."""
    c = Circuit(n)
    for i in reversed(range(n)):
        c.h[i]
        for k in range(i):
            c.cphase(math.pi / 2 ** (i - k))[k, i]
    for i in range(n // 2):
        c.swap[i, n - 1 - i]
    return c


def build_order_finding() -> Circuit:
    c = Circuit(N_WORK + N_COUNT)

    with c.block("order-finding(a=2, N=15)"):
        with c.block("init |x=1>"):
            c.x[0]

        with c.block("superposition"):
            c.h[COUNT[0], COUNT[1], COUNT[2]]

        # U: |x> -> |2x mod 15> is a cyclic left-shift of 4 bits.
        # U^(2^k) shifts by 2^k; U^4 = identity, so only k = 0, 1 matter.
        with c.block("c-U^1"):
            ctl = COUNT[0]
            c.cswap[ctl, 2, 3].cswap[ctl, 1, 2].cswap[ctl, 0, 1]

        with c.block("c-U^2"):
            ctl = COUNT[1]
            c.cswap[ctl, 0, 2].cswap[ctl, 1, 3]

        # IQFT on the counting register: reuse the QFT library circuit,
        # invert it with dagger(), and place it at offset 4.
        c.append_block("IQFT", qft_circuit(N_COUNT).dagger(), offset=N_WORK)

    return c


if __name__ == "__main__":
    print("Shor's order finding for N = 15, a = 2, with named blocks")
    print("=" * 60)

    c = build_order_finding()
    print(c.tree())
    print()

    # Read out the counting register's probabilities (differentiable API,
    # but here just used for exact classical post-processing).
    probs = c.probs(COUNT)
    peaks = {k: p.item() for k, p in enumerate(probs) if p.item() > 1e-9}
    print("Counting-register outcomes (value: probability):")
    for k, p in sorted(peaks.items()):
        print(f"  {k:3d} (phase {k}/8): {p:.4f}")

    # Phase peaks sit at multiples of 1/r with r = 4: 0, 2, 4, 6 out of 8.
    assert set(peaks) == {0, 2, 4, 6}, peaks
    assert all(abs(p - 0.25) < 1e-9 for p in peaks.values())

    # Recover the order r from the non-trivial phase 1/4, then the factors.
    r = 4
    assert pow(A, r, N) == 1
    f1, f2 = gcd(A ** (r // 2) - 1, N), gcd(A ** (r // 2) + 1, N)
    print(f"\nOrder r = {r};  gcd(a^(r/2) -+ 1, N) = {f1}, {f2}")
    assert {f1, f2} == {3, 5}
    print(f"Factors of {N}: {f1} x {f2}  -- OK")
