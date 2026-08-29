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
"""Decoders.

A decoder is anything with ``decode(detectors) -> int``, where `detectors` is
**the ids of the detectors that fired** -- not a bit string over all detectors.
An empty list means nothing fired. (Handing it a full bit string of zeros would
read as "detector 0 fired", which is the kind of quiet mistake an ambiguous
signature invites, so it is spelled out on every method here.)

Given those ids the decoder says whether the logical observable was flipped.
Keeping the interface that narrow is deliberate: swapping in another decoder,
or checking one against an exact reference, should not require touching the
experiment around it.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ['Decoder', 'MatchingDecoder', 'DetectorGraph']


class DetectorGraph:
    """Which errors flip which detectors, and which flip the logical observable.

    A node is a detector; the special node ``None`` is the boundary, where an
    error can terminate without another defect to pair with. Each edge carries
    the weight of the error that causes it and whether that error also flips the
    observable -- which is the only thing the decoder finally has to decide.
    """

    def __init__(self) -> None:
        self.edges: Dict[Tuple[Optional[int], Optional[int]], Tuple[float, bool]] = {}
        self.nodes: Set[int] = set()
        #: Faults that fire three or more detectors, which matching cannot
        #: represent. Counted rather than hidden: under circuit-level noise they
        #: are a real part of the model, and a decoder that ignores them is
        #: correspondingly imperfect.
        self.hyperedges: int = 0

    def add_error(self, detectors: Sequence[Optional[int]], weight: float = 1.0,
                  flips_observable: bool = False,
                  on_hyperedge: str = 'raise') -> None:
        """Record an error that fires `detectors` (one or two of them).

        `on_hyperedge` says what to do with a fault firing more than two:
        ``'raise'`` (the default) or ``'skip'``, which counts it in
        :attr:`hyperedges` and moves on.
        """
        ends = [d for d in detectors if d is not None]
        for node in ends:
            self.nodes.add(node)
        if len(ends) > 2:
            self.hyperedges += 1
            if on_hyperedge == 'skip':
                return
            raise ValueError("An error touching more than two detectors cannot be "
                             "matched; this decoder handles graph-like codes. Pass "
                             "on_hyperedge='skip' to count and ignore them.")
        a = ends[0] if ends else None
        b = ends[1] if len(ends) > 1 else None
        key = (a, b) if (b is None or (a is not None and a <= b)) else (b, a)
        previous = self.edges.get(key)
        if previous is None or weight < previous[0]:
            self.edges[key] = (weight, flips_observable)

    def shortest_paths(self) -> Dict[int, Dict[Optional[int], Tuple[float, bool]]]:
        """For every pair of detectors (and every detector to the boundary), the
        cheapest chain of errors joining them and whether it flips the observable.

        Plain Floyd-Warshall over a small graph -- codes worth decoding this way
        have detector counts in the thousands at most, and being obviously
        correct matters more here than being fast.
        """
        nodes: List[Optional[int]] = sorted(self.nodes) + [None]
        best: Dict[Optional[int], Dict[Optional[int], Tuple[float, bool]]] = {
            a: {b: (float('inf'), False) for b in nodes} for a in nodes}
        for node in nodes:
            best[node][node] = (0.0, False)
        for (a, b), (weight, flips) in self.edges.items():
            for x, y in ((a, b), (b, a)):
                if best[x][y][0] > weight:
                    best[x][y] = (weight, flips)
        for k in nodes:
            for i in nodes:
                if best[i][k][0] == float('inf'):
                    continue
                for j in nodes:
                    through = best[i][k][0] + best[k][j][0]
                    if through < best[i][j][0]:
                        best[i][j] = (through, best[i][k][1] ^ best[k][j][1])
        return best


class Decoder:
    """Interface: ``decode(fired_detector_ids) -> 0 or 1``."""

    def decode(self, detectors: Iterable[int]) -> int:
        """`detectors` is the ids of the detectors that fired (an empty
        iterable when none did), not a bit string over all detectors."""
        raise NotImplementedError


class MatchingDecoder(Decoder):
    """Minimum-weight perfect matching over the detector graph.

    Defects are paired with each other, or with the boundary, along the cheapest
    chains available; the predicted logical flip is the parity of how many of
    those chains cross the observable. This is the standard decoder for
    graph-like codes -- the repetition code and the surface code's two halves.

    The matching itself is exact (``networkx.max_weight_matching``), not greedy.
    """

    def __init__(self, graph: DetectorGraph) -> None:
        self.graph = graph
        self._paths = graph.shortest_paths()

    def decode(self, detectors: Iterable[int]) -> int:
        """`detectors` is the ids of the detectors that fired, not a bit string."""
        defects = sorted(set(detectors))
        if not defects:
            return 0
        import networkx as nx

        # Every defect may instead pair with the boundary. Giving each one its own
        # boundary copy, joined to the others at zero cost, turns "pair up or go to
        # the boundary" into an ordinary perfect matching.
        graph = nx.Graph()
        boundary = {d: ('boundary', d) for d in defects}
        for i, a in enumerate(defects):
            for b in defects[i + 1:]:
                weight = self._paths[a][b][0]
                if weight < float('inf'):
                    graph.add_edge(a, b, weight=-weight)
            to_edge = self._paths[a][None][0]
            if to_edge < float('inf'):
                graph.add_edge(a, boundary[a], weight=-to_edge)
        for i, a in enumerate(defects):
            for b in defects[i + 1:]:
                graph.add_edge(boundary[a], boundary[b], weight=0.0)

        matching = nx.max_weight_matching(graph, maxcardinality=True)
        flips = 0
        for u, v in matching:
            if isinstance(u, tuple) and isinstance(v, tuple):
                continue                      # boundary paired to boundary: no chain
            if isinstance(u, tuple):
                u, v = v, u
            other = None if isinstance(v, tuple) else v
            flips ^= int(self._paths[u][other][1])
        return flips
