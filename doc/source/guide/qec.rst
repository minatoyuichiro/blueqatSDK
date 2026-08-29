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
0.02     0.0077    0.0020    0.0010
0.05     0.0460    0.0330    0.0222
0.10     0.1445    0.1573    0.1737
0.20     0.3448    0.4175    0.4377
=======  ========  ========  ========

Below about 10% a longer code fails less; above it, longer fails more. That
crossing is the threshold, and for this code and noise model it should sit near
10.9%, where the equivalent random-bond Ising model orders.

The rotated surface code behaves the same way, ``rounds = d``, 1500 shots:

=======  ========  ========
``p``    ``d=3``   ``d=5``
=======  ========  ========
0.005    0.0020    0.0007
0.010    0.0120    0.0067
0.020    0.0367    0.0400
0.050    0.1680    0.2727
=======  ========  ========

The crossing here sits between 1% and 2%, below the ~3% this model is usually
quoted at. Two reasons, both worth knowing before reading a number off a run
like this: only two distances are being compared, so finite-size effects are
large, and the matching graph gives every error the same weight rather than
weighting by its likelihood.
