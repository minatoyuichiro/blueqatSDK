Circuits and gates
==================

Building circuits
-----------------

:class:`~blueqat.circuit.Circuit` stores a list of operations. Gates are
attributes; qubits are selected with ``[...]``; parametric gates take their
parameters as a call before the qubit indexing. Everything chains:

.. code-block:: python

   import math
   from blueqat import Circuit

   Circuit().h[0].cx[0, 1].rz(math.pi / 4)[1].m[:]

Qubit 0 is always the least-significant bit of the statevector index
(``'10'`` means qubit 1 is 1, qubit 0 is 0 -- the same convention as Qiskit's
``Statevector``).

Gate set
--------

Single-qubit gates
   ``i``, ``x``, ``y``, ``z``, ``h``, ``s``, ``sdg``, ``t``, ``tdg``, ``sx``,
   ``sxdg``, ``phase(theta)`` (aliases ``p``, ``r``), ``rx(theta)``,
   ``ry(theta)``, ``rz(theta)``, ``u(theta, phi, lam[, gamma])``,
   ``mat1(matrix)`` (arbitrary 2x2 unitary).

Two-qubit gates
   ``cx`` (alias ``cnot``), ``cy``, ``cz``, ``ch``, ``swap``, ``iswap``,
   ``iswapdg``, ``cphase(theta)`` (aliases ``cp``, ``cr``), ``crx``, ``cry``,
   ``crz``, ``cu(theta, phi, lam[, gamma])``, ``rxx(theta)``, ``ryy(theta)``,
   ``rzz(theta)``, ``zz``, ``zzdg``, ``exch(theta)`` (Heisenberg exchange
   pulse, see :doc:`exchange_only`).

Three-qubit gates
   ``ccx`` (alias ``toffoli``), ``ccz``, ``cswap``.

Other operations
   ``m`` / ``measure`` (optionally ``m(key="name")`` for keyed mid-circuit
   measurement), ``reset``, ``barrier``.

Gates that take no parameters raise ``ValueError`` if parameters are passed
(e.g. ``x(0.5)[0]`` is rejected rather than silently ignored).

Introspection
-------------

.. code-block:: python

   c = Circuit(3).h[:].cx[0, 1].cx[1, 2].m[:]
   c.n_qubits      # 3
   c.depth()       # 4  (parallel gates count once; barriers don't count)
   c.count_ops()   # Counter({'h': 3, 'cx': 2, 'measure': 3})

Measurement probabilities (differentiable, optionally marginalized onto
selected qubits) and Hamiltonian expectation values:

.. code-block:: python

   from blueqat.utils import Z

   Circuit(2).h[0].cx[0, 1].probs()          # tensor([0.5, 0., 0., 0.5])
   Circuit(2).h[0].cx[0, 1].probs([1])       # marginal of qubit 1
   Circuit(1).rx(0.4)[0].expect(1.0 * Z[0])  # <Z> = cos(0.4)

Any Pauli expression works as the observable, including sums:
``c.expect(1.0 * Z[0] * Z[1] - 0.5 * X[2])``, equivalently
``c.run(hamiltonian=...)``. The value is computed term by term over the
statevector rather than by building the Hamiltonian as a ``2**n x 2**n``
matrix, so it costs ``O(terms * 2**n)`` and stays usable well past the
~13 qubits at which the matrix form becomes impractical.

Pauli exponentials
------------------

:meth:`~blueqat.circuit.Circuit.exp_pauli` appends ``exp(-i * theta * P)``
for a Pauli product ``P``, the building block of Trotter steps and of most
chemistry and QAOA ansatz circuits. The operator is given as a mapping from
qubit index to Pauli letter, so it carries no bit-order ambiguity and sparse
products stay short:

.. code-block:: python

   Circuit().exp_pauli({0: 'X', 1: 'X', 2: 'Z', 3: 'Y'}, 0.3)  # exp(-0.3i XXZY)
   Circuit().exp_pauli({5: 'Z'}, t)                            # == rz(2t)[5]

Because ``P**2 == I``, this is exactly ``cos(theta) - i sin(theta) P``. The
convention (no factor of 1/2) is the same as
:meth:`~blueqat.utils.Term.get_time_evolution`, which builds the same
sequence from a ``Term``. ``theta`` may be a ``torch.Tensor``, so the
parameter stays differentiable; ``'I'`` entries are ignored.

Inverse circuits
----------------

:meth:`~blueqat.circuit.Circuit.dagger` returns the Hermitian conjugate
(gates reversed and conjugated). Measurement and reset have no inverse;
``dagger(ignore_measurement=True)`` drops them instead of raising:

.. code-block:: python

   c = Circuit(3)  # ... build ...
   identity = c + c.dagger()   # uncomputes back to |0...0>

OpenQASM 2.0
------------

.. code-block:: python

   qasm = Circuit(2).h[0].cx[0, 1].to_qasm()

   from blueqat.circuit_funcs import from_qasm
   c = from_qasm(qasm)

Matrices into circuits
----------------------

A single-qubit matrix goes straight in as ``mat1``. A two-qubit one is
decomposed by :func:`~blueqat.decompose.decompose_two_qubit`:

.. code-block:: python

   from blueqat.decompose import decompose_two_qubit

   c = decompose_two_qubit(matrix)                       # on qubits 0 and 1
   c = decompose_two_qubit(matrix, targets=(2, 5), n_qubits=6)

It is exact up to global phase. The route is the Cartan (KAK) factorization,

``U = phase * (A1 (x) A2) exp(i(a XX + b YY + c ZZ)) (A3 (x) A4)``,

with the interaction emitted as ``rxx``/``ryy``/``rzz`` -- three of them for a
general unitary, six CX once compiled. Canonical angles that vanish are left
out, so structured gates cost less without being special-cased: ``cx``, ``cz``,
``cy`` and ``ch`` each come back as one rotation, ``iswap`` as two, ``swap`` as
three.

.. note::

   Six CX is not the optimal three. Reaching that needs the canonical angles
   folded into the Weyl chamber with matching local corrections, which is not
   implemented. On hardware, where a CX budget of around fifteen is what
   decides whether a result survives, the difference is worth knowing before
   building a circuit out of many general two-qubit blocks.

Anything larger -- a general n-qubit unitary, or an isometry -- has no
synthesizer here. Let another toolchain decompose it and import the result as
QASM:

.. code-block:: python

   # In Qiskit: transpile to a basis blueqat's parser reads, then dump.
   #   qc = transpile(circuit, basis_gates=["u", "cx"])
   #   text = qasm2.dumps(qc)

   from blueqat.circuit_funcs.qasm_parser import from_qasm
   c = from_qasm(text)
   c.run(shots=200000, seed=1)

``u``, ``cx``, ``reset``, ``barrier`` and ``measure`` all survive the trip.
Measurement *keys* do not -- OpenQASM 2.0 has nowhere to record them -- so add
``m(key=...)`` on the blueqat side if the results need naming.

JSON serialization
------------------

Circuits round-trip through a versioned, JSON-compatible schema (this is also
the cloud submission wire format):

.. code-block:: python

   from blueqat.circuit_funcs.json_serializer import serialize, deserialize

   data = serialize(Circuit(2).h[0].cx[0, 1])
   c = deserialize(data)

Drawing
-------

``run(backend='draw')`` renders the circuit with matplotlib. Every registered
gate is drawable; unknown (user-registered) gates are omitted with a
``UserWarning``.

Named gate blocks
-----------------

Real algorithms are nests of subroutines -- Shor's order finding is
initialization, controlled modular multiplications and an inverse QFT, each
built from smaller pieces. Named blocks keep that structure in the circuit
object without changing execution (every backend transparently sees the
inner gates):

.. code-block:: python

   c = Circuit(7)
   with c.block("order-finding"):
       with c.block("superposition"):
           c.h[4, 5, 6]
       with c.block("c-U^1"):
           c.cswap[4, 2, 3].cswap[4, 1, 2].cswap[4, 0, 1]
       # place a library circuit as a block, shifted to qubits 4..6
       c.append_block("IQFT", qft_circuit(3).dagger(), offset=4)

   print(c.tree())
   # Circuit(7)
   # └─ order-finding
   #    ├─ superposition
   #    │  └─ h[4, 5, 6]
   #    ├─ c-U^1
   #    │  └─ ...
   #    └─ IQFT
   #       └─ ...

Blocks nest arbitrarily, show up in ``repr()`` and :meth:`~blueqat.circuit.Circuit.tree`,
and survive :meth:`~blueqat.circuit.Circuit.dagger` as mirrored blocks
(``"order-finding†"``). ``depth()`` / ``count_ops()`` count the contained
gates; ``flatten()`` / JSON serialization expand blocks into plain gates
(the flat wire format keeps no hierarchy). The circuit drawer renders a
block as a single labeled box spanning its qubits; a circuit wrapped
entirely in one block automatically descends so its child blocks appear as
boxes. Pass ``expand_blocks=n`` to open ``n`` levels of blocks, or
``expand_blocks=True`` to draw every inner gate. See
``examples/shor_15.py`` for a complete Shor-at-15 program written this way.

Ancilla qubits
--------------

.. code-block:: python

   c = Circuit(4).h[:]
   with c.ancilla() as a:        # allocates a fresh qubit
       c.cx[0, a[0]]
       c.cx[0, a[0]]
   # the ancilla is reset to |0> on exit (reset=True by default)

Macros and custom gates
-----------------------

Register a function as a circuit method, or a gate class into the gate set:

.. code-block:: python

   from blueqat import BlueqatGlobalSetting
   from blueqat.decorators import circuitmacro

   @circuitmacro
   def bell(c, a, b):
       return c.h[a].cx[a, b]

   Circuit(2).bell(0, 1)

   BlueqatGlobalSetting.register_gate('mygate', MyGateClass)
