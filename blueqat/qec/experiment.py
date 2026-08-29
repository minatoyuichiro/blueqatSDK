# Copyright 2019-2026 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Memory experiments: keep a logical qubit alive for a while, then ask whether
it survived.

The pieces are deliberately separable -- a code says what to measure, this
module runs it under noise and turns outcomes into detectors, and a decoder
turns detectors into a verdict. Swapping the decoder (or checking one against
an exact reference) touches nothing else.
"""

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..stabilizer import StabilizerSimulator
from .codes import StabilizerCode
from .decoders import DetectorGraph, MatchingDecoder

__all__ = ['CircuitLevelNoise', 'PhenomenologicalNoise', 'MemoryResult',
           'build_detector_graph', 'deterministic_stabilizers',
           'memory_experiment', 'round_operations']

#: An error location. ('pre', qubit, round) is a Pauli on that data qubit just
#: before the round; ('op', round, index) is a Pauli applied right after that
#: operation of the round; ('meas', stabilizer, round) flips what that
#: measurement reports.
Location = Tuple


class PhenomenologicalNoise:
    """Data errors between rounds and faulty syndrome measurements.

    `p_data` is the chance of an X on each data qubit between two rounds;
    `p_measure` is the chance that a syndrome bit is reported wrong. Both are
    per round. This is the model most threshold results are quoted in, and the
    one whose answer can still be reasoned about by hand.
    """

    def __init__(self, p_data: float = 0.0, p_measure: float = 0.0) -> None:
        for name, value in (('p_data', p_data), ('p_measure', p_measure)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        self.p_data = float(p_data)
        self.p_measure = float(p_measure)

    def sample(self, code: StabilizerCode, rounds: int,
               rng: random.Random) -> Dict:
        """Draw one shot's worth of errors."""
        errors: Dict = {}
        for t in range(rounds):
            if self.p_data:
                for q in range(code.n_data):
                    if rng.random() < self.p_data:
                        errors[('pre', q, t)] = 'X'
            if self.p_measure:
                for i in range(code.n_stabilizers):
                    if rng.random() < self.p_measure:
                        errors[('meas', i, t)] = True
        return errors

    def probability(self, location: Tuple, error, op=None) -> float:
        """How likely this particular fault is -- what an edge weight is made of."""
        kind = location[0]
        if kind == 'pre':
            return self.p_data
        if kind == 'meas':
            return self.p_measure
        return 0.0

    def __repr__(self) -> str:
        return f"PhenomenologicalNoise(p_data={self.p_data}, p_measure={self.p_measure})"


class CircuitLevelNoise:
    """A fault after every gate, and at every measurement and reset.

    `p1` is the chance of a random Pauli after a one-qubit gate; `p2` the chance
    of a random non-identity two-qubit Pauli after a two-qubit gate; `p_measure`
    the chance a measurement reports the wrong bit; `p_reset` the chance a reset
    leaves the qubit in ``|1>``; `p_idle` the chance of a Pauli on a data qubit
    between rounds.

    The difference from :class:`PhenomenologicalNoise` is where the faults live.
    A Pauli landing between an ancilla's two-qubit gates rides the rest of them
    out onto **several** data qubits -- a hook error -- so a single fault can
    become a weight-2 data error. Which faults do that depends on the order the
    ancilla touches its data qubits, which is why
    :func:`~blueqat.qec.syndrome_round` takes that order as an argument: on a
    surface code it decides whether the circuit-level distance is `d` or only
    ``(d + 1) // 2``.
    """

    def __init__(self, p1: float = 0.0, p2: float = 0.0, p_measure: float = 0.0,
                 p_reset: float = 0.0, p_idle: float = 0.0) -> None:
        for name, value in (('p1', p1), ('p2', p2), ('p_measure', p_measure),
                            ('p_reset', p_reset), ('p_idle', p_idle)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        self.p1 = float(p1)
        self.p2 = float(p2)
        self.p_measure = float(p_measure)
        self.p_reset = float(p_reset)
        self.p_idle = float(p_idle)

    @classmethod
    def uniform(cls, p: float) -> 'CircuitLevelNoise':
        """The usual quoted form: one rate everywhere."""
        return cls(p1=p, p2=p, p_measure=p, p_reset=p, p_idle=p)

    def sample(self, code: StabilizerCode, rounds: int,
               rng: random.Random, order: Optional[Callable] = None) -> Dict:
        errors: Dict = {}
        ops = round_operations(code, order)
        two_qubit_paulis = [(a, b) for a in 'IXYZ' for b in 'IXYZ'
                            if not (a == 'I' and b == 'I')]
        for t in range(rounds):
            if self.p_idle:
                for q in range(code.n_data):
                    if rng.random() < self.p_idle:
                        errors[('pre', q, t)] = rng.choice('XYZ')
            for k, op in enumerate(ops):
                kind, tag, qubits = op
                if kind == 'gate':
                    if len(qubits) == 1:
                        if self.p1 and rng.random() < self.p1:
                            errors[('op', t, k)] = {qubits[0]: rng.choice('XYZ')}
                    elif self.p2 and rng.random() < self.p2:
                        a, b = rng.choice(two_qubit_paulis)
                        chosen = {}
                        if a != 'I':
                            chosen[qubits[0]] = a
                        if b != 'I':
                            chosen[qubits[1]] = b
                        errors[('op', t, k)] = chosen
                elif kind == 'measure':
                    if self.p_measure and rng.random() < self.p_measure:
                        errors[('meas', tag, t)] = True
                elif self.p_reset and rng.random() < self.p_reset:
                    errors[('op', t, k)] = {qubits[0]: 'X'}
        return errors

    def probability(self, location: Tuple, error, op=None) -> float:
        """How likely this particular fault is.

        Each rate is split evenly over the Paulis it can produce, so the three
        one-qubit Paulis share `p1` and the fifteen two-qubit ones share `p2`.
        """
        kind = location[0]
        if kind == 'pre':
            return self.p_idle / 3.0
        if kind == 'meas':
            return self.p_measure
        if kind == 'op' and op is not None:
            if op[0] == 'reset':
                return self.p_reset
            if len(op[2]) == 1:
                return self.p1 / 3.0
            return self.p2 / 15.0
        return 0.0

    def __repr__(self) -> str:
        return (f"CircuitLevelNoise(p1={self.p1}, p2={self.p2}, "
                f"p_measure={self.p_measure}, p_reset={self.p_reset}, "
                f"p_idle={self.p_idle})")


class MemoryResult:
    """What a memory experiment found."""

    def __init__(self, code: StabilizerCode, rounds: int, shots: int,
                 failures: int) -> None:
        self.code = code
        self.rounds = rounds
        self.shots = shots
        self.failures = failures

    @property
    def logical_error_rate(self) -> float:
        return self.failures / self.shots if self.shots else 0.0

    def __repr__(self) -> str:
        return (f"MemoryResult({self.code.name}, rounds={self.rounds}, "
                f"shots={self.shots}, failures={self.failures}, "
                f"rate={self.logical_error_rate:.4g})")


def _observable_qubits(code: StabilizerCode) -> List[int]:
    """Data qubits whose Z outcome the logical observable is the parity of."""
    return [q for q, p in enumerate(code.logical_z[0]) if p != 'I']


def round_operations(code: StabilizerCode,
                     order: Optional[Callable] = None) -> List[Tuple]:
    """One round as an explicit sequence of operations.

    Every entry is a point a circuit-level error can attach to, numbered by its
    position, so a noise model and the detector-graph builder name the same
    places without either having to re-derive the schedule.
    """
    from .circuits import index_order
    order = order or index_order
    ops: List[Tuple] = []
    for i in range(code.n_stabilizers):
        ancilla = code.ancilla_of(i)
        support = dict(code.support(i))
        ops.append(('gate', 'h', (ancilla, )))
        for q in order(code, i):
            ops.append(('gate', {'X': 'cx', 'Z': 'cz', 'Y': 'cy'}[support[q]],
                        (ancilla, q)))
        ops.append(('gate', 'h', (ancilla, )))
        ops.append(('measure', i, (ancilla, )))
        ops.append(('reset', i, (ancilla, )))
    return ops


def _run_shot(code: StabilizerCode, rounds: int, errors: Dict,
              rng: Optional[random.Random] = None,
              order: Optional[Callable] = None) -> Tuple[List[List[int]], int]:
    """Run the rounds with the given errors injected; return syndromes and the
    measured logical observable.

    Errors are Pauli operators attached to named points: ``('pre', qubit, round)``
    just before a round, ``('op', round, index)`` right after that operation, and
    ``('meas', stabilizer, round)`` flipping what a measurement reports. On a
    stabilizer circuit that is exactly as general as letting faults happen during
    the gates, and it keeps every location addressable by the graph builder.
    """
    sim = StabilizerSimulator(code.n_qubits,
                              seed=rng.getrandbits(63) if rng else 0)
    ops = round_operations(code, order)
    syndromes: List[List[int]] = []
    # One extra, error-free round at the end plays the part of a perfect final
    # measurement: it is what makes the last layer of detectors meaningful.
    for t in range(rounds + 1):
        for q in range(code.n_data):
            pauli = errors.get(('pre', q, t))
            if pauli:
                sim.apply(pauli.lower() if isinstance(pauli, str) else 'x', (q, ))
        row = [0] * code.n_stabilizers
        for k, op in enumerate(ops):
            kind = op[0]
            if kind == 'gate':
                sim.apply(op[1], op[2])
            elif kind == 'measure':
                bit = sim.measure(op[2][0])
                if errors.get(('meas', op[1], t)):
                    bit ^= 1
                row[op[1]] = bit
            else:
                sim.reset(op[2][0])
            injected = errors.get(('op', t, k))
            if injected:
                for qubit, pauli in injected.items():
                    sim.apply(pauli.lower(), (qubit, ))
        syndromes.append(row)

    observable = 0
    for q in _observable_qubits(code):
        observable ^= sim.measure(q)
    return syndromes, observable


def deterministic_stabilizers(code: StabilizerCode) -> List[bool]:
    """Which stabilizers already have a known value in the starting state.

    A memory experiment starts every data qubit in ``|0>``, which is an
    eigenstate of a Z-only stabilizer but not of one containing an X or a Y. The
    first measurement of an X-type check is therefore a fair coin even with no
    errors at all, and comparing it against zero would make it fire every other
    shot -- so it is not a detector, and only its *change* from the next round
    onward is.
    """
    return [all(p in 'IZ' for p in stabilizer) for stabilizer in code.stabilizers]


def _detectors(syndromes: Sequence[Sequence[int]], code: StabilizerCode) -> List[int]:
    """Which detectors fired.

    A detector is a syndrome bit differing from the same bit in the previous
    round. In the first round there is no previous round: a stabilizer whose
    value the initial state fixes is compared against that value, and one whose
    value is random contributes no detector at all.
    """
    n_stabilizers = len(code.stabilizers)
    known = deterministic_stabilizers(code)
    fired = []
    for t, row in enumerate(syndromes):
        for i, bit in enumerate(row):
            if t == 0:
                if known[i] and bit:
                    fired.append(i)
                continue
            if bit ^ syndromes[t - 1][i]:
                fired.append(t * n_stabilizers + i)
    return fired


def build_detector_graph(code: StabilizerCode, rounds: int,
                         noise=None, circuit_level: bool = False,
                         order: Optional[Callable] = None) -> DetectorGraph:
    """Work out which errors fire which detectors, by trying each one.

    Rather than deriving the graph from a code's geometry -- which has to be
    redone, and re-checked, for every code -- each error location is injected on
    its own into an otherwise perfect run and the detectors it fires are read
    off. Errors compose linearly on a stabilizer circuit, so single-error
    responses are the whole story.

    Passing `noise` weights each edge by how likely it is,
    ``-log(p / (1 - p))``, instead of giving every fault the same weight. That
    matters once several different faults produce the same pair of detectors:
    matching should prefer the explanation that is actually more probable, and
    with flat weights it cannot.

    With `circuit_level`, the locations are every Pauli after every gate of the
    extraction circuit rather than just data and measurement faults. Some of
    those faults fire three or more detectors: matching cannot represent those,
    so they are counted in ``graph.hyperedges`` and left out rather than
    silently mangled. A decoder built on this graph is correspondingly imperfect
    for them, which is the ordinary situation for matching decoders under
    circuit-level noise.
    """
    graph = DetectorGraph()
    baseline_syndromes, baseline_observable = _run_shot(code, rounds, {}, order=order)
    baseline = set(_detectors(baseline_syndromes, code))
    if baseline:
        raise ValueError("The error-free run already fires detectors; the code's "
                         "initial state is not in its own codespace.")

    if circuit_level and isinstance(noise, PhenomenologicalNoise):
        raise ValueError("circuit_level=True needs a CircuitLevelNoise to weight "
                         "gate faults; a phenomenological model has no rate for them.")

    ops = round_operations(code, order)
    # Keyed by (detectors, flips_observable): distinct faults can fire the same
    # detectors and yet disagree about the observable, and which of them a chain
    # really was is decided by which is more likely -- not by whichever the
    # enumeration happened to reach first.
    found: Dict[Tuple, float] = {}
    for location, error in _error_locations(code, rounds, circuit_level, order):
        syndromes, observable = _run_shot(code, rounds, {location: error}, order=order)
        fired = tuple(_detectors(syndromes, code))
        flips = bool(observable ^ baseline_observable)
        if not fired and not flips:
            continue          # invisible and harmless: nothing to decode
        probability = 1.0
        if noise is not None:
            op = ops[location[2]] if location[0] == 'op' else None
            probability = noise.probability(location, error, op)
            if probability <= 0.0:
                continue
        key = (fired, flips)
        found[key] = found.get(key, 0.0) + probability

    totals: Dict[Tuple, float] = {}
    dominant: Dict[Tuple, bool] = {}
    for (fired, flips), probability in found.items():
        totals[fired] = totals.get(fired, 0.0) + probability
        if probability > found.get((fired, not flips), 0.0):
            dominant[fired] = flips
        elif fired not in dominant:
            dominant[fired] = flips

    for fired, probability in totals.items():
        weight = 1.0
        if noise is not None:
            probability = min(max(probability, 1e-15), 0.5 - 1e-12)
            weight = -math.log(probability / (1.0 - probability))
        graph.add_error(fired, weight=weight, flips_observable=dominant[fired],
                        on_hyperedge='skip')
    return graph


def _error_locations(code: StabilizerCode, rounds: int, circuit_level: bool,
                     order: Optional[Callable]) -> List[Tuple]:
    """Every fault the graph builder should try, as ``(location, error)`` pairs.

    Exactly the faults a noise model can actually produce, and no others.

    The last round stands in for a perfect final readout: nothing goes wrong in
    it, and there is no gap before it for a data qubit to decay in. So faults
    live in rounds ``0 .. rounds-1`` only, and the last layer of detectors is
    reached in combination rather than by any single fault.

    Enumerating faults the sampler cannot generate is not harmless. They compete
    for the same edges, and since an edge's observable attribution goes to
    whichever mechanism is most likely, a fault that can never happen can carry
    an edge the wrong way.
    """
    out: List[Tuple] = []
    for t in range(rounds):
        out += [(('pre', q, t), 'X') for q in range(code.n_data)]
        out += [(('meas', i, t), True) for i in range(code.n_stabilizers)]
    if not circuit_level:
        return out

    ops = round_operations(code, order)
    for t in range(rounds):
        for k, op in enumerate(ops):
            kind, qubits = op[0], op[2]
            if kind == 'measure':
                # A faulty measurement reports the wrong bit; that is the
                # ('meas', ...) location above, not a Pauli left behind here.
                continue
            if kind == 'reset':
                # A failed reset leaves the qubit in |1>, which is an X and
                # nothing else: Z would do nothing to |0> and Y is that same X
                # up to a phase. Enumerating all three would credit each with
                # the full reset rate and treble it.
                out.append((('op', t, k), {qubits[0]: 'X'}))
            elif len(qubits) == 1:
                out += [(('op', t, k), {qubits[0]: p}) for p in 'XYZ']
            else:
                a, b = qubits
                out += [(('op', t, k), {a: pa, b: pb})
                        for pa in 'IXYZ' for pb in 'IXYZ'
                        if not (pa == 'I' and pb == 'I')]
    return out


def memory_experiment(code: StabilizerCode, rounds: int, noise, shots: int,
                      decoder=None, seed: Optional[int] = None,
                      order: Optional[Callable] = None) -> MemoryResult:
    """Hold a logical ``|0>`` for `rounds` rounds and count how often it is lost.

    A failure is a shot where the decoder's verdict disagrees with what actually
    happened to the observable -- not merely a shot with errors in it, which is
    the distinction the whole exercise is about.

    `noise` is a :class:`PhenomenologicalNoise` or a :class:`CircuitLevelNoise`;
    the default decoder is matched to whichever it is. `order` is the ancilla's
    interaction order, which only circuit-level noise can feel.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}.")
    if shots < 1:
        raise ValueError(f"shots must be at least 1, got {shots}.")
    circuit_level = isinstance(noise, CircuitLevelNoise)
    if decoder is None:
        decoder = MatchingDecoder(
            build_detector_graph(code, rounds, noise=noise,
                                 circuit_level=circuit_level, order=order))
    rng = random.Random(seed)

    failures = 0
    for _ in range(shots):
        errors = (noise.sample(code, rounds, rng, order) if circuit_level
                  else noise.sample(code, rounds, rng))
        syndromes, observable = _run_shot(code, rounds, errors, rng, order)
        prediction = decoder.decode(_detectors(syndromes, code))
        if prediction != observable:
            failures += 1
    return MemoryResult(code, rounds, shots, failures)
