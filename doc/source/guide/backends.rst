Backends and execution
======================

Simulation modes
----------------

One simulator, two execution modes:

- ``tensornet`` (default): tensor-network contraction via ``opt_einsum``.
  Never materializes the full state unless asked to, so wide-but-shallow
  circuits scale far beyond dense simulation.
- ``statevector``: dense statevector propagation.

.. code-block:: python

   Circuit(20).h[:].run()                      # tensornet (default)
   Circuit(20).h[:].run(backend='statevector') # dense
   Circuit(20).h[:].run(mode='statevector')    # equivalent

Both modes agree numerically and both preserve autograd graphs.

Return values
-------------

.. code-block:: python

   c = Circuit(2).h[0].cx[0, 1]

   c.run()                                   # statevector (torch.Tensor)
   c.statevector()                           # same, explicit
   c.m[:].run(shots=100)                     # Counter of bitstrings
   c.shots(100)                              # same, explicit
   c.run(amplitude='11')                     # a single amplitude
   c.m[:].oneshot()                          # (collapsed state, one outcome)
   c.expect(hamiltonian)                     # <psi|H|psi>
   c.probs([1])                              # marginal probabilities

Large circuits
--------------

The dense state has ``2**n`` entries. In ``tensornet`` mode, circuits with
more than 28 qubits require ``shots=`` or ``returns='amplitude'`` instead of
the full vector:

.. code-block:: python

   Circuit(50).h[:].run(shots=3)
   Circuit(50).h[:].run(returns='amplitude', amplitude='0' * 50)

Sampling uses inverse-CDF search, so there is no category-count limit.

Noise applies to every entry point
----------------------------------

``noise=`` (and ``quasi_static=``, ``noise_scale=``) selects the density-matrix
backend, whichever way the run is written:

.. code-block:: python

   c.run(noise=nm, shots=1000)
   c.shots(1000, noise=nm)        # the same thing
   c.probs(noise=nm)              # probabilities from the density matrix

``statevector()`` and ``oneshot()`` raise instead: a noisy state has no
statevector, and returning the noiseless one would be worse than an error.

Mid-circuit measurement
-----------------------

A measurement collapses the state, so anything acting on that qubit afterwards
sees a classical bit. Circuits where that happens are run shot by shot as
quantum trajectories, collapsing where the measurement is, and the reported bit
is the one the measurement produced -- not a value drawn from the final state
after later gates have moved it. Measurements that nothing follows keep the
fast single-pass path.

Reproducible sampling
---------------------

``seed=`` fixes every random draw a run makes -- shot sampling, mid-circuit
collapse and large-``n`` perfect sampling alike -- so the same circuit and
seed always give the same counts:

.. code-block:: python

   c = Circuit(4).h[:]
   c.run(shots=200, seed=42) == c.run(shots=200, seed=42)   # True

The seed drives a private ``torch.Generator``, not ``torch.manual_seed``, so
seeding a circuit does not disturb the RNG the rest of your program uses.
Without ``seed=``, runs stay random exactly as before.

Estimating from counts
----------------------

A ratio of two small counts is not an estimate. With 30000 trials and zero
failures, the failure rate is not 0 -- it is "below roughly 1e-4", and code
that divides by it produces either 0 or a meaningless spike depending on which
side of the fraction the zero lands on. Before dividing, decide a minimum count
and say so:

.. code-block:: python

   failures = counts.get('1', 0)
   if failures < 10:
       rate = None          # not enough events to estimate a ratio
   else:
       rate = failures / shots

The same applies to comparing two runs: a ratio of two rates each built from a
handful of events carries essentially no information, however many total shots
were taken.

Counts bit order
----------------

By default the leftmost character of a counts key is the highest-numbered
qubit, so ``key[-1]`` is qubit 0. Cloud APIs (including
``qapi.blueqat.app``) use the opposite layout; ``bit_order='q0_first'``
returns keys in that order instead of leaving each caller to reverse them by
hand:

.. code-block:: python

   Circuit(3).x[0].run(shots=4)                          # Counter({'001': 4})
   Circuit(3).x[0].run(shots=4, bit_order='q0_first')    # Counter({'100': 4})

Keys are always zero-padded to exactly ``n_qubits`` characters, in both
orders -- an unpadded ``'11'`` would be ambiguous between qubits 0 and 1 and
qubits 4 and 5 once reversed. :func:`~blueqat.backends.backendbase.apply_bit_order`
applies the same conversion to a ``Counter`` obtained elsewhere.

Mid-circuit measurement and reset
---------------------------------

``reset`` and keyed measurement make outcomes depend on when the collapse
happens, so such circuits automatically run shot-by-shot as quantum
trajectories, collapsing at each ``measure`` / ``reset``:

.. code-block:: python

   Circuit(2).h[0].cx[0, 1].reset[0].m[:].run(shots=100)

   Circuit().x[0].m(key='a')[0].run(shots=3, returns='samples')
   # [{'a': [1]}, {'a': [1]}, {'a': [1]}]

Custom initial states
---------------------

.. code-block:: python

   import torch
   psi0 = torch.tensor([0, 1, 0, 0], dtype=torch.complex128)
   Circuit(2).h[0].run(initial=psi0)

Other built-in backends
-----------------------

- ``'draw'`` -- matplotlib circuit diagram.
- ``'draw_tn'`` -- the tensor-network graph of the circuit.
- ``'eo'`` -- exchange-only transpiler (see :doc:`exchange_only`).
- ``'cloud'`` -- cloud submission (see :doc:`cloud`).
- ``'1q_compaction'`` / ``'2q_decomposition'`` -- transpilers merging
  single-qubit gates / rewriting two-qubit gates into a chosen basis.

Registering your own backend
----------------------------

.. code-block:: python

   from blueqat import register_backend, Backend

   class MyBackend(Backend):
       def run(self, gates, n_qubits, *args, **kwargs):
           ...

   register_backend('mybackend', MyBackend)
   Circuit(2).h[0].run(backend='mybackend')
   Circuit(2).h[0].run_with_mybackend()      # equivalent
