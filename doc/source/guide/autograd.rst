Differentiable circuits, VQE and QAOA
=====================================

Gradients through the simulator
-------------------------------

Any gate parameter may be a :class:`torch.Tensor` with
``requires_grad=True``. The whole pipeline -- gate matrices, state
propagation (in both execution modes), probabilities, expectation values --
is built from differentiable torch operations:

.. code-block:: python

   import torch
   from blueqat import Circuit
   from blueqat.utils import Z

   theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
   energy = Circuit(1).rx(theta)[0].expect(1.0 * Z[0])
   energy.backward()
   theta.grad        # -sin(0.4), the exact analytic gradient

This means variational algorithms need no parameter-shift rule: plain
``torch.optim`` optimizers work directly.

Pauli operators and Hamiltonians
--------------------------------

:mod:`blueqat.utils` provides the Pauli algebra:

.. code-block:: python

   from blueqat.utils import X, Y, Z, I, from_qubo, qubo_bit

   h = 0.5 * Z[0] * Z[1] + 1.2 * X[0] - 3.0
   h = h.simplify()
   h.to_matrix(2)                   # dense or sparse torch matrix
   term = (X[0] * Y[1]).to_term()
   evo = term.get_time_evolution()  # appends exp(-i t P) to a circuit

``from_qubo`` converts a QUBO cost matrix into an Ising Hamiltonian.

VQE
---

.. code-block:: python

   import torch
   from blueqat import Circuit
   from blueqat.utils import AnsatzBase, Vqe, Z, X

   class MyAnsatz(AnsatzBase):
       def get_circuit(self, params):
           return Circuit(2).rx(params[0])[0].ry(params[1])[1].cx[0, 1]

   hamiltonian = (1.0 * Z[0] * Z[1] + 0.5 * X[0]).simplify()
   ansatz = MyAnsatz(hamiltonian, n_params=2)
   result = Vqe(ansatz).run()
   result.most_common(4)

``Vqe`` accepts any ``torch.optim`` optimizer class, an optional sampler
(e.g. ``get_measurement_sampler(n)`` for shot-based estimation or
``non_sampling_sampler`` for exact, gradient-preserving expectation), and
``initial_params``.

Reproducible runs and convergence
---------------------------------

Without ``initial_params``, ``Vqe.run()`` starts from random parameters, so
repeated runs of the same problem land in different local optima -- for a
QAOA instance that can mean a wildly different probability of finding the
optimum from one run to the next. ``seed=`` pins the whole run down:

.. code-block:: python

   Vqe(ansatz, seed=42).run()          # or: Vqe(ansatz).run(seed=42)

One seed covers both sources of randomness: the initial parameters, and the
sampler's draws when it is a seedable one built by
``get_measurement_sampler(n, seed=...)``. Like ``Circuit.run(seed=...)`` it
uses a private generator, leaving the global RNG alone.

Every run records its objective value at each iteration, so convergence can
be inspected without re-running under another optimizer:

.. code-block:: python

   result = Vqe(ansatz, seed=42).run()
   len(result.loss_history)      # iterations actually taken
   result.loss_history[-1]       # last recorded objective value

Shot noise and the parameter-shift rule
---------------------------------------

Estimating an expectation value from shots throws away the autograd graph -- a
count is a number, not a differentiable function of the gate angles -- so a
shot-based objective has no gradient to backpropagate. ``Vqe`` notices and
switches to the **parameter-shift rule**, which recovers the gradient from the
same estimator by evaluating it at shifted parameters:

.. code-block:: python

   from blueqat.utils import get_measurement_sampler

   vqe = Vqe(ansatz, sampler=get_measurement_sampler(2000, seed=3), seed=42)
   result = vqe.run()          # works; backpropagation alone cannot

Each gate contributes ``(E(theta + pi/2) - E(theta - pi/2)) / 2``, which is
exact rather than a finite-difference approximation, and those are chained onto
the ansatz parameters through autograd -- so a parameter driving many gates, as
QAOA's angles do, correctly sums their contributions.

``gradient=`` overrides the choice: ``'backprop'`` always differentiates
(and fails on a shot sampler), ``'parameter_shift'`` always uses the rule, and
the default ``'auto'`` picks by whether the objective came back differentiable.
The rule costs two extra circuit evaluations per parametric gate application, so
it is not a free replacement for backpropagation on the exact path.

It is exact only for gates whose generator has two eigenvalues one apart:
``rx``, ``ry``, ``rz``, ``p``/``phase``, ``rxx``, ``ryy``, ``rzz``, ``cp`` and
``exch``. Controlled rotations (``crx``, ``cry``, ``crz``) have four and would
need a four-term rule, so an ansatz containing them raises rather than being
given a quietly wrong gradient.

:func:`~blueqat.utils.parameter_shift_gradient` exposes the same machinery
directly, returning the energy and gradient for any ansatz and energy estimator.

How many shots?
~~~~~~~~~~~~~~~

Fewer than one might expect. Optimizing a 27-parameter QAOA instance (Max-Cut
on K6, ``step=1``, 120 iterations) at 100, 1000 and 8000 shots per estimate all
converged, reaching the optimal cut with probability 0.66 to 0.72 -- no
degradation at the low end, so whatever threshold exists lies below 100.

Wall-clock time barely moved across those three levels either, which is the
useful thing to know: the cost of a shot-based run is dominated by *how many
circuit evaluations the shift rule asks for* -- two per parametric gate
application per iteration -- not by the shots inside each one. Trading shots
away to go faster mostly does not work; reducing parameters or iterations does.

Shot noise also behaves a little like the noise in stochastic gradient descent.
In that same experiment the exact-gradient reference happened to settle into a
poor local optimum while every shot-based run found a good one. That is one
configuration and not a general rule -- the same problem reaches a good optimum
from most starting points anyway -- but it is a reason not to assume a noisy
gradient is simply a degraded one.

QAOA
----

:class:`~blueqat.utils.QaoaAnsatz` builds the standard QAOA ansatz from a
Hamiltonian whose terms must mutually commute (checked automatically):

.. code-block:: python

   from blueqat.utils import QaoaAnsatz, Vqe, from_qubo

   qubo = [[1, 1], [1, 0]]
   h = from_qubo(qubo)
   ansatz = QaoaAnsatz(h.simplify(), step=2)
   result = Vqe(ansatz).run()
   print(result.most_common(2))

See ``examples/maxcut_qaoa.py`` and ``examples/vqe_ground_state.py`` in the
repository for complete, self-verifying programs.
