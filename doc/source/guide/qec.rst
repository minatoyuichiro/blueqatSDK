Error correction
================

:mod:`blueqat.qec` keeps four things apart: a **code** says what to measure, a
**circuit** says how, a **decoder** says what the outcomes meant, and an
**experiment** puts them together. Each can be replaced without the others
noticing -- which is what makes a decoder checkable against a reference.

.. code-block:: python

   from blueqat.qec import repetition_code, memory_experiment, PhenomenologicalNoise

   code = repetition_code(5)
   result = memory_experiment(code, rounds=5, shots=2000, seed=1,
                              noise=PhenomenologicalNoise(p_data=0.02, p_measure=0.02))
   result.logical_error_rate

Codes
-----

:func:`~blueqat.qec.repetition_code` and
:func:`~blueqat.qec.rotated_surface_code` return a
:class:`~blueqat.qec.StabilizerCode`, which carries the stabilizer generators,
the logical operators, and the data/ancilla layout. Pauli strings are indexed by
qubit -- character `q` acts on data qubit `q`.

``code.check()`` raises unless the generators really commute and the logicals
really pair up, and ``code.logical_weight()`` brute-forces the code distance for
small codes. Both exist because a layout mistake otherwise surfaces only as a
quietly wrong threshold:

.. code-block:: python

   rotated_surface_code(3).logical_weight()   # 3
   repetition_code(3).logical_weight()        # 1 -- it stops bit flips only

Syndrome circuits
-----------------

:func:`~blueqat.qec.syndrome_round` measures every stabilizer with its own
ancilla, prepared in ``|+>``, coupled by a controlled Pauli, rotated back and
measured. Measurements are keyed ``"s{index}_r{round}"`` so rounds stay
distinguishable.

The interaction order is ascending data-qubit index and is **documented rather
than assumed**, because on a surface code that order decides which two-qubit
errors propagate into weight-2 data errors -- hook errors -- and so decides
whether the circuit-level distance is `d` or only ``(d+1)/2``. A schedule
chosen for that reason belongs to the experiment, so it is passed in:

.. code-block:: python

   syndrome_round(code, order=my_schedule)   # my_schedule(code, index) -> qubits

What counts as a detector
-------------------------

A detector is a syndrome bit differing from the same bit in the previous round.
The first round has no previous round, and there the initial state matters: all
data qubits start in ``|0>``, which fixes every Z-only stabilizer at ``+1`` but
leaves an X-type check a **fair coin even with no errors at all**. So a Z-type
check's first outcome is compared against its known value, and an X-type
check's first outcome is not a detector -- only its change from the second
round onward is. :func:`~blueqat.qec.deterministic_stabilizers` says which is
which.

Getting this wrong is not subtle in its symptoms: the error-free run starts
firing detectors half the time, which
:func:`~blueqat.qec.build_detector_graph` refuses outright rather than
building a graph around it.

Decoders
--------

A decoder is anything with ``decode(detectors) -> 0 or 1``, where `detectors`
is **the ids of the detectors that fired** -- not a bit string over all of
them. An empty list means nothing fired.
:class:`~blueqat.qec.MatchingDecoder` does exact minimum-weight perfect matching
over a :class:`~blueqat.qec.DetectorGraph`.

That graph is not written out per code. :func:`~blueqat.qec.build_detector_graph`
injects each error location on its own into an otherwise perfect run and reads
off which detectors fire and whether the observable flips -- errors compose
linearly on a stabilizer circuit, so single-error responses are the whole story.
A new code therefore needs no new geometry code, and no new chance to get the
geometry wrong.

Thresholds
----------

A memory experiment holds a logical ``|0>`` for some rounds and counts the shots
where the decoder's verdict disagrees with what actually happened. Sweeping the
physical error rate and the distance shows the threshold directly -- measured
here on the repetition code, ``rounds = d``, 4000 shots:

=======  ========  ========  ========
``p``    ``d=3``   ``d=5``   ``d=7``
=======  ========  ========  ========
0.02     0.0077    0.0018    0.0005
0.05     0.0455    0.0283    0.0163
0.10     0.1378    0.1398    0.1495
0.20     0.3360    0.4005    0.4400
=======  ========  ========  ========

Below about 10% a longer code fails less; above it, longer fails more. That
crossing is the threshold, and for this code and noise model it should sit near
10.9%, where the equivalent random-bond Ising model orders.

The rotated surface code behaves the same way, ``rounds = d``, 1500 shots:

=======  ========  ========
``p``    ``d=3``   ``d=5``
=======  ========  ========
0.005    0.0013    0.0007
0.010    0.0087    0.0053
0.020    0.0280    0.0333
0.050    0.1600    0.2453
=======  ========  ========

The crossing here sits just below 2%, against the ~3% this model is usually
quoted at. Only two distances are being compared, so finite-size effects are
large -- worth knowing before reading a number off a run like this.

Circuit-level noise
-------------------

:class:`~blueqat.qec.CircuitLevelNoise` puts faults where they actually happen:
after every gate, at every measurement, at every reset, and on idle data
qubits.

.. code-block:: python

   from blueqat.qec import CircuitLevelNoise

   memory_experiment(code, rounds=3, shots=6000, seed=4,
                     noise=CircuitLevelNoise.uniform(0.005))
   # or, closer to hardware, a two-qubit rate an order of magnitude larger:
   CircuitLevelNoise(p1=0.001, p2=0.01, p_measure=0.01)

The difference from the phenomenological model is not just "more places to go
wrong". A fault landing between an ancilla's two-qubit gates rides the rest of
them out onto **several** data qubits -- a hook error -- so one fault can
become a weight-2 data error. Which faults do that depends on the order the
ancilla visits its data qubits, and that order is an argument precisely so the
question can be asked. Measured on the surface code, ``d=3``, ``p=0.003``,
3000 shots:

=========================  ====================
Interaction order          Logical error rate
=========================  ====================
ascending index            0.0197
reversed                   0.0167
``0,2,1,3``                0.0103
=========================  ====================

Nearly a factor of two from nothing but the order. Enumerating every single
fault instead of sampling shows why, and splits the damage in two. Some faults
the matching graph *can* represent are still decoded wrongly, because two
faults share a detector signature and disagree about the observable: no decoder
reading only that signature can tell them apart, and their product is an
undetectable logical error. The rest fire three or more detectors, which
matching has no edge for at all; those are counted in ``graph.hyperedges`` and
left out rather than mangled into one. Over all 1299 single faults, the
ascending order is decoded wrongly 76 times and ``0,2,1,3`` 38 times.

That second group is the ordinary state of a matching decoder under
circuit-level noise, not something to paper over -- a hypergraph decoder would
recover them.

The time boundary
~~~~~~~~~~~~~~~~~

Some failures are neither, and no interaction order removes them. In the last
detector layer, a fault on an ancilla and a fault on a boundary data qubit can
light the *same single detector*: one flips the observable and the other does
not, and the syndrome cannot tell them apart. Every round but the last is
checked twice in time -- by itself and by the round after -- and the last one is
not, so the effective distance halves there. Enumerating all 1299 single faults
on the ``d=3`` surface code, thirteen such collisions survive under
``0,2,1,3``; they are present under the ascending order too, alongside the
order-dependent ones.

More rounds dilute their share, and a different treatment of the final layer
avoids them. Changing the gate order does not, because they do not come from
gate order.

Edge weights
------------

Passing the noise model to :func:`~blueqat.qec.build_detector_graph` weights
each edge by ``-log(p / (1 - p))`` instead of giving every fault the same
weight, so matching prefers the more probable explanation when several faults
fire the same detectors.

It is worth saying what that bought here, which was nothing measurable. On the
repetition code at ``d = 3, 5, 7``, both under a uniform rate and under a
two-qubit rate ten times the one-qubit rate, weighted and flat decoding agreed
to within the shot noise of 12000 shots -- sometimes one ahead, sometimes the
other. The weights are principled and the machinery is there for models where
the rates differ more sharply; on this evidence they are not what stands
between these thresholds and the textbook ones.

What the weights *do* decide is which explanation wins when two faults fire the
same detectors and disagree about the observable. That makes it essential that
the graph enumerate exactly the faults the noise model can produce and no
others: a fault the sampler will never generate still gets a vote, and can
carry an edge the wrong way.

It also means **a decoder belongs to the rate it was built at**. Weights are
``-log(p / (1 - p))``, so changing `p` moves them by different amounts, and one
explanation can overtake another. On the ``d=3`` surface code the detector pair
``{12, 16}`` is close enough to such a crossover to fall on either side of it:

=========  =============  ================  ============================
``p``      Direct edge    Two boundaries    Cheaper explanation
=========  =============  ================  ============================
0.005      8.0060342      8.7055254         the direct edge, by 0.6995
0.010      7.3125535      7.2908808         the boundaries, by 0.0217
=========  =============  ================  ============================

The single fault that actually joins ``12`` and ``16`` does not cross the
observable; going out to the boundary twice does. So the same syndrome decodes
correctly at ``p = 0.005`` and wrongly at ``p = 0.01``, purely because the
weights reordered.

Neither answer is a bug -- minimum-weight decoding is optimal on average, not
fault by fault -- but it does mean two runs whose graphs were built at
different rates are not comparable, however identical everything else looks.
:func:`~blueqat.qec.memory_experiment` builds its decoder from the noise it is
given, so it is consistent by construction; a decoder cached across a sweep of
`p` is not.
