Circuit optimization
====================

:func:`~blueqat.optimize.optimize` applies three peephole rewrites to a fixed
point: drop gates that are the identity, cancel adjacent inverses, and merge
adjacent rotations about the same axis.

.. code-block:: python

   from blueqat import Circuit
   from blueqat.optimize import optimize

   optimize(Circuit(2).h[0].x[1].h[0])              # Circuit(2).x[1]
   optimize(Circuit(2).rz(0.3)[0].x[1].rz(0.4)[0])  # rz(0.7)[0] . x[1]
   optimize(Circuit(2).cx[0, 1].cx[0, 1])           # empty

"Adjacent" means adjacent *on the qubits involved*: the two ``h`` gates above
cancel even though an ``x`` on the other qubit sits between them.

What it will not do
-------------------

Every rewrite preserves the unitary exactly, **global phase included**. A
rotation is therefore dropped only at a multiple of its true identity period --
``4*pi`` for ``rx``, ``ry``, ``rz`` and the two-qubit Pauli rotations, ``2*pi``
for ``p`` and ``exch``. At ``2*pi`` the first group equals ``-I``, and removing
one would silently flip the sign of a statevector.

Gates are matched as ordered targets unless the gate is symmetric in them:
``cz[0, 1]`` cancels ``cz[1, 0]``, but ``cx[0, 1]`` does not cancel
``cx[1, 0]``, because that pair is not the identity.

A rotation whose angle is a ``torch.Tensor`` with ``requires_grad`` is never
dropped, even at zero: its value may be zero while its gradient is not, so
removing the gate would change what an optimizer sees rather than just
shortening the circuit. Merging such rotations is fine and keeps them
differentiable.

Barriers, measurements and resets are never removed, and nothing is reordered
across them.

Blocks and slices are expanded first (as :func:`~blueqat.circuit_funcs.flatten`
does), since the rewrites work on individual gate applications.

Exchange-only circuits
----------------------

For exchange-only spin qubits the cost that matters is the **pulse count**, and
optimizing pays twice -- once on the logical circuit, where whole pulse
sequences vanish before they are ever emitted, and once on the pulses
themselves, where consecutive pulses on the same spin pair fuse:

.. code-block:: python

   import blueqat.eo
   from blueqat.optimize import optimize

   logical = Circuit(2).x[0].x[0].cx[0, 1].cx[0, 1].h[1]

   len(logical.run(backend='eo').ops)              # 65 pulses
   len(optimize(logical).run(backend='eo').ops)    # 3

============================  ========  ==================
Logical circuit               Direct    Optimized first
============================  ========  ==================
``x x cx cx h``               65        3
``s s s s``                   4         0  (pulse stage)
``rz rz rz``                  3         1
``h h``                       6         0
============================  ========  ==================

Note that optimizing the *pulses* cannot undo a logically trivial sequence:
``h h`` is the identity on the encoded qubit, but its six pulses are not the
identity on the full physical space, so only the logical stage removes them.
That is why the logical pass runs first.
