Noise and density matrices
==========================

Passing ``noise=`` to :meth:`~blueqat.circuit.Circuit.run` switches the circuit
onto a density-matrix simulation and applies a channel after every gate:

.. code-block:: python

   from blueqat import Circuit
   from blueqat.noise import depolarizing

   rho = Circuit(2).h[0].cx[0, 1].run(noise=depolarizing(0.01))

Without ``noise=`` the density-matrix backend is still reachable as
``run(backend='density')``, which returns ``|psi><psi|`` for the pure state.

Channels
--------

.. code-block:: python

   from blueqat.noise import (depolarizing, pauli_depolarizing,
                              amplitude_damping, phase_damping, kraus)

   depolarizing(p)          # (1-p) rho + p I / 2**k
   pauli_depolarizing(p)    # (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)
   amplitude_damping(gamma) # decay of |1> towards |0>   (T1)
   phase_damping(lam)       # loss of coherence, no energy loss  (T2)
   kraus([k0, k1, ...])     # any channel, from explicit Kraus operators

:func:`~blueqat.noise.depolarizing` is the Nielsen & Chuang definition: with
probability ``p`` the state is replaced by the maximally mixed state. The other
convention in circulation reads ``p`` as the probability that some Pauli error
occurred; that is :func:`~blueqat.noise.pauli_depolarizing`, and the two agree
when ``p_pauli = 3 * p / 4``. Getting this wrong shifts every number, so the two
are kept as separate names rather than as one argument.

After a two-qubit gate, depolarizing acts on both of that gate's qubits
**jointly** by default -- a mixture over all ``4**k`` Pauli strings. With
``per_qubit=True`` the single-qubit channel is applied to each of the gate's
qubits independently instead:

.. code-block:: python

   depolarizing(0.02)                   # joint two-qubit channel after a cx
   depolarizing(0.02, per_qubit=True)   # one-qubit channel on each of its qubits

These are genuinely different maps and both are wanted: the joint one is the
k-qubit channel as usually written, while a paper assuming purely *local* noise
means the independent one. At the same rate the local form damps a bit harder,
because two channels touch the state where one did.

Damping channels are single-qubit and are applied to each qubit a gate touched.
Measurement, reset and barrier never carry noise.

Noise models
------------

A bare channel applies after every gate. To give different gates different
rates -- real devices have two-qubit errors an order of magnitude larger --
name them:

.. code-block:: python

   from blueqat.noise import NoiseModel, depolarizing, amplitude_damping

   nm = NoiseModel()
   nm.add(depolarizing(0.001))                     # after every gate
   nm.add(depolarizing(0.01), gates=['cx', 'cz'])  # ...and more after these
   nm.add(amplitude_damping(0.002))

   Circuit(3).h[0].cx[0, 1].run(noise=nm)

``noise=`` also accepts a list of channels, applied in order.

Quasi-static noise
------------------

Silicon spin qubits are not dephased mainly by a Markovian channel. Nuclear
(Overhauser) fields and 1/f charge noise drift far more slowly than a circuit
runs, so each repetition sees an essentially **constant** detuning and the
average over repetitions is what decoheres. That correlation in time is not
something Kraus operators can express, and the difference is measurable rather
than academic:

.. code-block:: python

   from blueqat.noise import QuasiStatic

   Circuit(1).h[0].i[0].i[0].run(quasi_static=QuasiStatic(sigma=0.4),
                                 samples=4000, seed=1)

Each sample freezes one detuning per qubit, drawn from ``N(0, sigma)``, and
accumulates ``rz(delta_q * dt)`` on every qubit after each layer of the
circuit; the resulting density matrices are averaged, which is exactly the
classical mixture over detunings. Free induction decay then comes out Gaussian,
``exp(-(sigma * t)**2 / 2)`` with `t` counted in layers.

The sharp test is a **Hahn echo**: a flip in the middle of the wait undoes a
static offset accumulated before it, and does nothing at all to a memoryless
channel. Measured coherence, same circuit and same total wait:

=========================  ==============  =============
Noise                      Without echo    With echo
=========================  ==============  =============
``QuasiStatic(0.4)``       0.02            **0.92**
``phase_damping(0.25)``    0.32            0.27
=========================  ==============  =============

So a T2* or echo experiment reproduced with :func:`~blueqat.noise.phase_damping`
will give the wrong answer no matter how the rate is tuned.

``samples`` sets how many detunings are averaged (200 by default); the error
falls as ``1/sqrt(samples)``. Quasi-static noise composes with channels -- pass
both ``quasi_static=`` and ``noise=``.

Scaling the noise
-----------------

``noise_scale=c`` multiplies every channel's rate by ``c``. This is the knob
zero-noise extrapolation turns: run the same circuit at several noise levels
and extrapolate the expectation value back to zero.

.. code-block:: python

   import numpy as np
   from blueqat.utils import Z

   c = Circuit(2).h[0].cx[0, 1]
   h = 1.0 * Z[0] * Z[1]
   scales = [1.0, 2.0, 3.0]
   values = [float(c.run(noise=depolarizing(0.02), noise_scale=s, hamiltonian=h))
             for s in scales]
   zero_noise = np.polyfit(scales, values, 1)[-1]   # extrapolate to scale 0

A scale that would take a rate outside its valid range raises rather than
being silently clipped, since a clipped point would quietly corrupt the fit.

``noise_scale`` reaches quasi-static noise too, but scales ``sigma`` by
``sqrt(c)`` rather than by ``c``: Gaussian dephasing decays as
``exp(-(sigma t)**2 / 2)``, so it is ``sigma**2`` that the extrapolation is
linear in.

Results from a noisy run
------------------------

.. code-block:: python

   c.run(noise=nm)                      # the density matrix, a 2**n x 2**n tensor
   c.run(noise=nm, shots=1000, seed=1)  # counts sampled from its diagonal
   c.run(noise=nm, hamiltonian=h)       # Tr(rho H)

``shots`` accepts the same ``seed`` and ``bit_order`` arguments as the
statevector backends. ``returns='statevector'``, ``'amplitude'`` and
``'samples'`` are statevector notions and are refused here.

Cost
----

A density matrix has ``4**n`` entries and every gate touches all of them, so
this backend is for small circuits. A gate and the channels following it are
multiplied into a single operator and applied in one pass, which is about eight
times faster than applying each Kraus operator separately, but the scaling is
what it is:

=========  ==================  ==============
Qubits     Density matrix      Per gate
=========  ==================  ==============
8          1 MB                0.4 ms
10         17 MB               9 ms
12         268 MB              183 ms
=========  ==================  ==============

Comfortable to about 10 qubits, usable to 12; above 14 the backend refuses
rather than exhausting memory.
