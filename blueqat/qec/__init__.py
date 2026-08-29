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
"""Quantum error correction: codes, syndrome circuits, decoders, experiments.

    from blueqat.qec import repetition_code, memory_experiment, PhenomenologicalNoise

    code = repetition_code(5)
    result = memory_experiment(code, rounds=5, shots=2000, seed=1,
                               noise=PhenomenologicalNoise(p_data=0.02, p_measure=0.02))
    result.logical_error_rate

The four pieces are kept apart on purpose: a code says what to measure, a
circuit says how, a decoder says what the outcomes meant, and an experiment
puts them together. Each can be replaced without the others noticing.

A decoder's ``decode(detectors)`` takes **the ids of the detectors that fired**,
not a bit string over all of them.
"""

from .codes import StabilizerCode, repetition_code, rotated_surface_code
from .circuits import index_order, syndrome_extraction_circuit, syndrome_round
from .decoders import Decoder, DetectorGraph, MatchingDecoder
from .experiment import (CircuitLevelNoise, MemoryResult, PhenomenologicalNoise,
                         build_detector_graph, deterministic_stabilizers,
                         memory_experiment, round_operations)

__all__ = [
    'StabilizerCode', 'repetition_code', 'rotated_surface_code',
    'index_order', 'syndrome_extraction_circuit', 'syndrome_round',
    'Decoder', 'DetectorGraph', 'MatchingDecoder',
    'CircuitLevelNoise', 'MemoryResult', 'PhenomenologicalNoise',
    'build_detector_graph', 'deterministic_stabilizers', 'memory_experiment',
    'round_operations',
]
