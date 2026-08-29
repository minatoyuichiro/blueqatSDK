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
"""Error correction: codes, syndrome circuits, decoders and memory experiments."""

import pytest

from blueqat import Circuit
from blueqat.qec import (DetectorGraph, MatchingDecoder, PhenomenologicalNoise,
                         StabilizerCode, build_detector_graph, index_order,
                         memory_experiment, repetition_code,
                         rotated_surface_code, syndrome_extraction_circuit,
                         syndrome_round)


# ---------------------------------------------------------------- the codes

@pytest.mark.parametrize('distance', [3, 5, 7])
def test_repetition_code_is_a_valid_code(distance):
    code = repetition_code(distance)
    code.check()                                  # raises if generators clash
    assert code.n_data == distance
    assert code.n_stabilizers == distance - 1
    assert code.n_qubits == 2 * distance - 1      # data plus one ancilla each


@pytest.mark.parametrize('distance', [3, 5])
def test_rotated_surface_code_is_a_valid_code(distance):
    code = rotated_surface_code(distance)
    code.check()
    assert code.n_data == distance * distance
    assert code.n_stabilizers == distance * distance - 1


def test_rotated_surface_code_has_the_distance_it_claims():
    # Brute force over every Pauli: a layout mistake shows up as a logical
    # operator lighter than d, which is the failure that would otherwise only
    # appear as a quietly wrong threshold.
    assert rotated_surface_code(3).logical_weight() == 3


def test_repetition_code_protects_only_against_bit_flips():
    # Its lightest logical is a single Z, which is the honest statement of what
    # a repetition code does -- and why it is a test bed, not a real code.
    assert repetition_code(3).logical_weight() == 1


def test_a_broken_code_is_rejected():
    # Two stabilizers that do not commute.
    bad = StabilizerCode('bad', 2, ['XI', 'ZI'], ['XX'], ['ZZ'])
    with pytest.raises(ValueError, match='commute'):
        bad.check()


def test_pauli_strings_must_match_the_qubit_count():
    with pytest.raises(ValueError, match='length'):
        StabilizerCode('bad', 3, ['ZZ'], ['XXX'], ['ZII'])


def test_bad_distances_are_refused():
    with pytest.raises(ValueError):
        repetition_code(1)
    with pytest.raises(ValueError):
        rotated_surface_code(4)       # even distances are not rotated codes


# ------------------------------------------------------- syndrome circuits

def test_syndrome_round_measures_every_stabilizer():
    code = repetition_code(3)
    circuit = syndrome_round(code)
    keys = [op.key for op in circuit.ops if op.lowername == 'measure']
    assert keys == ['s0_r0', 's1_r0']
    assert circuit.n_qubits == code.n_qubits


def test_syndrome_round_uses_the_documented_interaction_order():
    code = repetition_code(3)
    circuit = syndrome_round(code)
    pairs = [tuple(op.targets) for op in circuit.ops if op.lowername == 'cz']
    # Ancilla 3 checks qubits 0,1; ancilla 4 checks 1,2 -- ascending data index.
    assert pairs == [(3, 0), (3, 1), (4, 1), (4, 2)]


def test_interaction_order_can_be_overridden():
    code = repetition_code(3)
    reversed_order = lambda c, i: list(reversed(index_order(c, i)))
    pairs = [tuple(op.targets)
             for op in syndrome_round(code, order=reversed_order).ops
             if op.lowername == 'cz']
    assert pairs == [(3, 1), (3, 0), (4, 2), (4, 1)]


def test_an_order_naming_the_wrong_qubit_is_refused():
    code = repetition_code(3)
    with pytest.raises(ValueError, match='does not act on'):
        syndrome_round(code, order=lambda c, i: [2, 0])


def test_multiple_rounds_are_keyed_apart():
    circuit = syndrome_extraction_circuit(repetition_code(3), rounds=3)
    keys = [op.key for op in circuit.ops if op.lowername == 'measure']
    assert keys == ['s0_r0', 's1_r0', 's0_r1', 's1_r1', 's0_r2', 's1_r2']


def test_a_clean_round_reports_no_syndrome():
    code = repetition_code(3)
    counts = syndrome_round(code).run(backend='stabilizer', shots=8, seed=1)
    # Ancillas 3 and 4 are reset, so every reported bit is 0.
    assert set(counts) == {'0' * code.n_qubits}


def test_an_injected_error_lights_the_neighbouring_checks():
    code = repetition_code(3)
    circuit = Circuit(code.n_qubits).x[1] + syndrome_round(code, reset_ancillas=False)
    counts = circuit.run(backend='stabilizer', shots=4, seed=1)
    key, = counts
    # X on data qubit 1 sits between both checks, so both ancillas read 1.
    assert key[::-1][3] == '1' and key[::-1][4] == '1'


# ------------------------------------------------------------- the decoder

def test_matching_decoder_on_a_hand_built_graph():
    graph = DetectorGraph()
    graph.add_error([0], flips_observable=True)      # error crossing the observable
    graph.add_error([0, 1])
    graph.add_error([1])
    decoder = MatchingDecoder(graph)
    assert decoder.decode([]) == 0
    assert decoder.decode([0]) == 1
    assert decoder.decode([1]) == 0
    assert decoder.decode([0, 1]) == 0


def test_detector_graph_is_built_from_injected_errors():
    graph = build_detector_graph(repetition_code(3), rounds=1)
    # Two stabilizers over two detector layers.
    assert graph.nodes <= {0, 1, 2, 3}
    assert graph.edges


def test_a_hyperedge_is_refused():
    graph = DetectorGraph()
    with pytest.raises(ValueError, match='more than two'):
        graph.add_error([0, 1, 2])


# --------------------------------------------------------- the experiment

def test_a_noiseless_memory_never_fails():
    result = memory_experiment(repetition_code(3), rounds=3, shots=100, seed=1,
                               noise=PhenomenologicalNoise())
    assert result.failures == 0
    assert result.logical_error_rate == 0.0


def test_below_threshold_a_bigger_code_fails_less():
    # The point of the whole exercise: at a low enough physical error rate,
    # distance buys protection.
    noise = PhenomenologicalNoise(p_data=0.02, p_measure=0.02)
    small = memory_experiment(repetition_code(3), rounds=3, noise=noise,
                              shots=2000, seed=7)
    large = memory_experiment(repetition_code(7), rounds=7, noise=noise,
                              shots=2000, seed=7)
    assert large.logical_error_rate < small.logical_error_rate / 2


def test_above_threshold_a_bigger_code_fails_more():
    # ...and above it, distance actively hurts, which is what makes the
    # crossing a threshold rather than a trend.
    noise = PhenomenologicalNoise(p_data=0.2, p_measure=0.2)
    small = memory_experiment(repetition_code(3), rounds=3, noise=noise,
                              shots=1000, seed=7)
    large = memory_experiment(repetition_code(7), rounds=7, noise=noise,
                              shots=1000, seed=7)
    assert large.logical_error_rate > small.logical_error_rate


def test_the_experiment_is_reproducible():
    noise = PhenomenologicalNoise(p_data=0.05, p_measure=0.05)
    kwargs = dict(rounds=3, noise=noise, shots=300)
    a = memory_experiment(repetition_code(3), seed=11, **kwargs)
    b = memory_experiment(repetition_code(3), seed=11, **kwargs)
    c = memory_experiment(repetition_code(3), seed=12, **kwargs)
    assert a.failures == b.failures
    assert a.failures != c.failures


def test_a_decoder_can_be_supplied():
    code = repetition_code(3)
    decoder = MatchingDecoder(build_detector_graph(code, rounds=2))
    result = memory_experiment(code, rounds=2, shots=200, seed=1, decoder=decoder,
                               noise=PhenomenologicalNoise(0.05, 0.05))
    assert 0.0 <= result.logical_error_rate <= 1.0


def test_a_decoder_that_always_says_no_flip_does_worse():
    # The comparison a swappable decoder is for.
    class Never:
        def decode(self, detectors):
            return 0

    noise = PhenomenologicalNoise(p_data=0.05, p_measure=0.05)
    kwargs = dict(rounds=5, noise=noise, shots=1000, seed=3)
    matched = memory_experiment(repetition_code(5), **kwargs)
    naive = memory_experiment(repetition_code(5), decoder=Never(), **kwargs)
    assert matched.logical_error_rate < naive.logical_error_rate


# ------------------------------------------------- the surface code memory

def test_x_checks_are_not_detectors_in_the_first_round():
    # |0...0> fixes the Z-type checks but not the X-type ones, so a first-round
    # X outcome is a fair coin even with no errors -- it cannot be a detector.
    from blueqat.qec import deterministic_stabilizers
    code = rotated_surface_code(3)
    known = deterministic_stabilizers(code)
    assert sum(known) == 4 and len(known) == 8
    assert all(deterministic_stabilizers(repetition_code(5)))


def test_the_surface_code_detector_graph_builds():
    # Regression: this used to raise "the error-free run already fires
    # detectors", because the random first-round X outcomes were being compared
    # against zero.
    graph = build_detector_graph(rotated_surface_code(3), rounds=3)
    assert graph.nodes and graph.edges


def test_a_noiseless_surface_code_memory_never_fails():
    result = memory_experiment(rotated_surface_code(3), rounds=3, shots=60, seed=1,
                               noise=PhenomenologicalNoise())
    assert result.failures == 0


def test_the_surface_code_gains_from_distance_below_threshold():
    noise = PhenomenologicalNoise(p_data=0.005, p_measure=0.005)
    small = memory_experiment(rotated_surface_code(3), rounds=3, noise=noise,
                              shots=1500, seed=5)
    large = memory_experiment(rotated_surface_code(5), rounds=5, noise=noise,
                              shots=1500, seed=5)
    assert large.logical_error_rate < small.logical_error_rate


def test_bad_experiment_parameters_are_refused():
    code = repetition_code(3)
    with pytest.raises(ValueError):
        memory_experiment(code, rounds=0, noise=PhenomenologicalNoise(), shots=10)
    with pytest.raises(ValueError):
        memory_experiment(code, rounds=1, noise=PhenomenologicalNoise(), shots=0)
    with pytest.raises(ValueError):
        PhenomenologicalNoise(p_data=1.5)
