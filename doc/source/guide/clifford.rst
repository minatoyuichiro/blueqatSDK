Clifford operators
==================

:class:`~blueqat.clifford.Clifford` stores an operator as a stabilizer tableau
-- the images of ``X_0..X_{n-1}`` and ``Z_0..Z_{n-1}`` under conjugation --
rather than as a matrix. Composition and inversion are then exact bit
operations, with no ``2**n`` array anywhere.

.. code-block:: python

   from blueqat import Circuit
   from blueqat.clifford import Clifford, random_clifford

   c = Clifford.from_circuit(Circuit(2).h[0].cx[0, 1])
   c.to_circuit()          # back to gates: h, s, sdg, cx, x, z
   c.inverse()
   c.then(other)           # apply c first, then other

The Clifford gate set is ``i``, ``x``, ``y``, ``z``, ``h``, ``s``, ``sdg``,
``sx``, ``sxdg``, ``cx``, ``cy``, ``cz`` and ``swap``. Anything else -- a ``t``
or an ``rx``, say -- raises rather than being silently approximated. Global
phase is not tracked: it is unobservable, and the Clifford group as
benchmarking uses it is defined modulo phase.

Uniform random Cliffords
------------------------

.. code-block:: python

   random_clifford(2, seed=0)     # uniform over the 11520 two-qubit Cliffords

Uniformity comes from building a random symplectic basis one conjugate pair at
a time: the image of ``X_i`` is drawn uniformly from the non-identity Paulis
still available, the image of ``Z_i`` uniformly from those anticommuting with
it, and the rest of the operator from what commutes with both. Those counts
multiply to ``|Sp(2n, 2)|`` exactly, and ``2n`` independent sign bits supply the
remaining Pauli factor. ``seed`` uses a private generator, so it does not
disturb the global RNG.

Randomized benchmarking
-----------------------

The reason the tableau matters: a benchmarking sequence has to end with the
*single* Clifford that undoes everything before it. Replaying the sequence
backwards would roughly double its length and measure something else.

.. code-block:: python

   from blueqat import Circuit
   from blueqat.clifford import Clifford, random_clifford
   from blueqat.noise import depolarizing

   def rb_circuit(n, m, seed):
       total = Clifford.identity(n)
       circuit = Circuit(n)
       for i in range(m):
           c = random_clifford(n, seed=seed * 1000 + i)
           circuit += c.to_circuit()
           total = total.then(c)
       return circuit + total.inverse().to_circuit()

   # Survival probability: 1 exactly without noise, decaying with it
   rho = rb_circuit(1, 16, seed=0).run(noise=depolarizing(0.02))
   survival = float(rho[0, 0].real)

Fitting ``survival`` against sequence length ``m`` is the benchmarking analysis
itself, which lives with whoever is doing the experiment rather than in the SDK.
