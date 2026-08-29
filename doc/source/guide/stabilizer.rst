Stabilizer simulation
=====================

The ``'stabilizer'`` backend stores a state by its stabilizer generators
instead of its amplitudes, so memory is ``O(n**2)`` bits rather than ``2**n``
complex numbers:

.. code-block:: python

   from blueqat import Circuit

   circuit = Circuit(200).h[0]
   for q in range(199):
       circuit.cx[q, q + 1]
   circuit.m[:].run(backend='stabilizer', shots=20, seed=1)
   # => Counter({'000...0': 11, '111...1': 9})

The price is Gottesman-Knill: only Clifford gates (``i``, ``x``, ``y``, ``z``,
``h``, ``s``, ``sdg``, ``sx``, ``sxdg``, ``cx``, ``cy``, ``cz``, ``swap``),
measurement and reset. A ``t`` or an ``rx`` raises -- use the statevector or
density-matrix backend for those.

This is what makes error-correction work possible: a distance-5 surface code
needs 49 qubits before any noise is added, which is far past what a statevector
(and much further past what a density matrix) can hold.

Inspecting the state
--------------------

Without ``shots``, the run returns the simulator itself:

.. code-block:: python

   sim = Circuit(3).h[0].cx[0, 1].cx[1, 2].run(backend='stabilizer')
   sim.stabilizers()      # ['+XXX', '+ZZI', '+IZZ']
   sim.measure(0)         # collapses, returns 0 or 1
   sim.reset(0)
   sim.copy()             # branch the state

Character `q` of a stabilizer string is the Pauli on qubit `q`.

``shots`` takes the same ``seed`` and ``bit_order`` as the other backends. Each
shot is an independent trajectory, since measurement outcomes are genuinely
random and there is no final state to sample from afterwards.

Cost
----

A measurement touches every row, and each row is ``O(n)`` bits, so a full
measurement pass is ``O(n**3 / 64)`` word operations per shot. Measured on a
GHZ chain, 20 shots: 0.3 s at 100 qubits, 7 s at 500, 119 s at 2000. Circuits
that measure only a few qubits are correspondingly cheaper.
