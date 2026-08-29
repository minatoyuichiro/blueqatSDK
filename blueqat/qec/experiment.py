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

import random
from typing import Dict, List, Optional, Sequence, Tuple

from ..stabilizer import StabilizerSimulator
from .codes import StabilizerCode
from .decoders import DetectorGraph, MatchingDecoder

__all__ = ['PhenomenologicalNoise', 'MemoryResult', 'build_detector_graph',
           'deterministic_stabilizers', 'memory_experiment']

#: An error location: ('data', qubit, round) is an X on that data qubit just
#: before that round's measurement; ('meas', stabilizer, round) flips what that
#: measurement reports.
Location = Tuple[str, int, int]


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
               rng: random.Random) -> Dict[Location, bool]:
        """Draw one shot's worth of errors."""
        errors: Dict[Location, bool] = {}
        for t in range(rounds):
            if self.p_data:
                for q in range(code.n_data):
                    if rng.random() < self.p_data:
                        errors[('data', q, t)] = True
            if self.p_measure:
                for i in range(code.n_stabilizers):
                    if rng.random() < self.p_measure:
                        errors[('meas', i, t)] = True
        return errors

    def __repr__(self) -> str:
        return f"PhenomenologicalNoise(p_data={self.p_data}, p_measure={self.p_measure})"


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


def _run_shot(code: StabilizerCode, rounds: int, errors: Dict[Location, bool],
              rng: Optional[random.Random] = None) -> Tuple[List[List[int]], int]:
    """Run the rounds with the given errors injected; return syndromes and the
    measured logical observable.

    Gates themselves are ideal here: the noise model injects Pauli errors at the
    named locations, which for a stabilizer circuit is exactly equivalent to
    letting them happen during the gates, and keeps every location addressable
    by the detector-graph builder.
    """
    sim = StabilizerSimulator(code.n_qubits,
                              seed=rng.getrandbits(63) if rng else 0)
    syndromes: List[List[int]] = []
    # One extra, error-free round at the end plays the part of a perfect final
    # measurement: it is what makes the last layer of detectors meaningful.
    for t in range(rounds + 1):
        for q in range(code.n_data):
            if errors.get(('data', q, t)):
                sim.apply('x', (q, ))
        row = []
        for i in range(code.n_stabilizers):
            ancilla = code.ancilla_of(i)
            sim.apply('h', (ancilla, ))
            for q, pauli in code.support(i):
                sim.apply({'X': 'cx', 'Z': 'cz', 'Y': 'cy'}[pauli], (ancilla, q))
            sim.apply('h', (ancilla, ))
            bit = sim.measure(ancilla)
            if errors.get(('meas', i, t)):
                bit ^= 1
            row.append(bit)
            sim.reset(ancilla)
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


def build_detector_graph(code: StabilizerCode, rounds: int) -> DetectorGraph:
    """Work out which errors fire which detectors, by trying each one.

    Rather than deriving the graph from a code's geometry -- which has to be
    redone, and re-checked, for every code -- each error location is injected on
    its own into an otherwise perfect run and the detectors it fires are read
    off. Errors compose linearly on a stabilizer circuit, so single-error
    responses are the whole story.
    """
    graph = DetectorGraph()
    baseline_syndromes, baseline_observable = _run_shot(code, rounds, {})
    baseline = set(_detectors(baseline_syndromes, code))
    if baseline:
        raise ValueError("The error-free run already fires detectors; the code's "
                         "initial state is not in its own codespace.")

    locations: List[Location] = []
    for t in range(rounds + 1):
        locations += [('data', q, t) for q in range(code.n_data)]
        if t < rounds + 1:
            locations += [('meas', i, t) for i in range(code.n_stabilizers)]

    for location in locations:
        syndromes, observable = _run_shot(code, rounds, {location: True})
        fired = _detectors(syndromes, code)
        if not fired and observable == baseline_observable:
            continue          # invisible and harmless: nothing to decode
        graph.add_error(fired, weight=1.0,
                        flips_observable=bool(observable ^ baseline_observable))
    return graph


def memory_experiment(code: StabilizerCode, rounds: int,
                      noise: PhenomenologicalNoise, shots: int,
                      decoder=None, seed: Optional[int] = None) -> MemoryResult:
    """Hold a logical ``|0>`` for `rounds` rounds and count how often it is lost.

    A failure is a shot where the decoder's verdict disagrees with what actually
    happened to the observable -- not merely a shot with errors in it, which is
    the distinction the whole exercise is about.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}.")
    if shots < 1:
        raise ValueError(f"shots must be at least 1, got {shots}.")
    if decoder is None:
        decoder = MatchingDecoder(build_detector_graph(code, rounds))
    rng = random.Random(seed)

    failures = 0
    for _ in range(shots):
        errors = noise.sample(code, rounds, rng)
        syndromes, observable = _run_shot(code, rounds, errors, rng)
        prediction = decoder.decode(_detectors(syndromes, code))
        if prediction != observable:
            failures += 1
    return MemoryResult(code, rounds, shots, failures)
