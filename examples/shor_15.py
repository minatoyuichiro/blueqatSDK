"""Shor's algorithm (order finding for N = 15, a = 2) as a deep nest of blocks.

Shor's algorithm is a hierarchy of subroutines, and this example keeps every
level of that hierarchy in the circuit itself:

    Shor order-finding (N=15, a=2)
    ├─ init |x=1>
    ├─ superposition (counting)
    ├─ modular exponentiation
    │  ├─ c-U^1 (x2 mod 15)
    │  └─ c-U^2 (x4 mod 15)
    └─ IQFT (counting)
       ├─ bit reversal†
       └─ QFT stage q_k† ...

`print(c.tree())` shows the whole nest, and the drawer zooms through it:
`expand_blocks=1` draws the four phase-estimation stages as boxes,
`expand_blocks=2` opens modular exponentiation and the IQFT one level more,
and `expand_blocks=True` shows every gate. Execution is untouched by any of
this -- the assertions at the bottom check the physics end to end.

U|x> = |2x mod 15> is a cyclic left-shift of the 4-bit work register, so the
controlled powers are short cswap chains; U^4 = identity, which is why the
order r = 4 appears as phase peaks at multiples of 1/4.
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
    """QFT on qubits 0..n-1, with each textbook stage as its own block."""
    c = Circuit(n)
    for i in reversed(range(n)):
        with c.block(f"QFT stage q{i}"):
            c.h[i]
            for k in range(i):
                c.cphase(math.pi / 2 ** (i - k))[k, i]
    with c.block("bit reversal"):
        for i in range(n // 2):
            c.swap[i, n - 1 - i]
    return c


def build_order_finding() -> Circuit:
    c = Circuit(N_WORK + N_COUNT)

    with c.block(f"Shor order-finding (N={N}, a={A})"):
        with c.block("init |x=1>"):
            c.x[0]

        with c.block("superposition (counting)"):
            c.h[COUNT[0], COUNT[1], COUNT[2]]

        with c.block("modular exponentiation"):
            with c.block("c-U^1 (x2 mod 15)"):
                ctl = COUNT[0]
                c.cswap[ctl, 2, 3].cswap[ctl, 1, 2].cswap[ctl, 0, 1]
            with c.block("c-U^2 (x4 mod 15)"):
                ctl = COUNT[1]
                c.cswap[ctl, 0, 2].cswap[ctl, 1, 3]
            # c-U^4 = identity: order 4 divides 2^2.

        # The IQFT is the daggered QFT library circuit, placed on the
        # counting register. dagger() mirrors the inner blocks (stage†),
        # and append_block's offset shift preserves that structure.
        c.append_block("IQFT (counting)", qft_circuit(N_COUNT).dagger(),
                       offset=N_WORK)

    return c


if __name__ == "__main__":
    print("Shor's order finding for N = 15, a = 2, with nested blocks")
    print("=" * 60)

    c = build_order_finding()
    print(c.tree())
    print()

    # The drawer zooms through the hierarchy (uncomment in a notebook):
    # c.run(backend="draw")                    # 4 top-level stage boxes
    # c.run(backend="draw", expand_blocks=1)   # same (auto-descends the wrapper)
    # c.run(backend="draw", expand_blocks=2)   # opens modexp + IQFT stages
    # c.run(backend="draw", expand_blocks=True)  # every gate

    probs = c.probs(COUNT)
    peaks = {k: p.item() for k, p in enumerate(probs) if p.item() > 1e-9}
    print("Counting-register outcomes (value: probability):")
    for k, p in sorted(peaks.items()):
        print(f"  {k:3d} (phase {k}/8): {p:.4f}")

    # Phase peaks sit at multiples of 1/r with r = 4: 0, 2, 4, 6 out of 8.
    assert set(peaks) == {0, 2, 4, 6}, peaks
    assert all(abs(p - 0.25) < 1e-9 for p in peaks.values())

    r = 4
    assert pow(A, r, N) == 1
    f1, f2 = gcd(A ** (r // 2) - 1, N), gcd(A ** (r // 2) + 1, N)
    print(f"\nOrder r = {r};  gcd(a^(r/2) -+ 1, N) = {f1}, {f2}")
    assert {f1, f2} == {3, 5}
    print(f"Factors of {N}: {f1} x {f2}  -- OK")
