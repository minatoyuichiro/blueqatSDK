# Copyright 2019 The Blueqat Developers
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

from collections import Counter
import pytest
import torch
from blueqat.utils import to_inttuple

@pytest.mark.parametrize('arg, expect', [
    ("01011", (0, 1, 0, 1, 1)),
    ({"00011": 2, "10100": 3}, {(0, 0, 0, 1, 1): 2, (1, 0, 1, 0, 0): 3}),
    (Counter({"00011": 2, "10100": 3}), Counter({(0, 0, 0, 1, 1): 2, (1, 0, 1, 0, 0): 3}))
])
def test_to_inttuple(arg, expect):
    assert to_inttuple(arg) == expect


# --------------------------------------------------------- random_unitary

def test_random_unitary_is_unitary():
    import torch
    from blueqat.utils import check_unitarity, random_unitary
    for dim in (2, 4, 8):
        assert check_unitarity(random_unitary(dim, seed=dim))


def test_random_unitary_is_reproducible():
    import torch
    from blueqat.utils import random_unitary
    assert torch.allclose(random_unitary(4, seed=3), random_unitary(4, seed=3))
    assert not torch.allclose(random_unitary(4, seed=3), random_unitary(4, seed=4))


def test_random_unitary_does_not_disturb_the_global_rng():
    import torch
    from blueqat.utils import random_unitary
    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    random_unitary(8, seed=99)
    assert torch.allclose(torch.rand(3), expected)


def test_random_unitary_phases_are_uniform():
    """The QR trap, made visible.

    Plain `torch.linalg.qr` returns a Q whose column *magnitudes* are already
    Haar distributed -- so the usual heavy-output-probability check passes even
    without the fix -- but whose entry *phases* are strongly concentrated.
    Multiplying by the phases of R's diagonal is what spreads them out.
    """
    import cmath
    import torch
    from blueqat.utils import random_unitary

    def concentration(angles):
        # |mean of e^{i*angle}|: 0 for a uniform phase, 1 for a fixed one.
        return abs(sum(cmath.exp(1j * a) for a in angles)) / len(angles)

    fixed = [cmath.phase(complex(random_unitary(4, seed=2000 + t)[0, 0]))
             for t in range(1500)]
    assert concentration(fixed) < 0.1

    generator = torch.Generator()
    generator.manual_seed(7)
    naive = []
    for _ in range(1500):
        a = torch.complex(
            torch.randn(4, 4, dtype=torch.float64, generator=generator),
            torch.randn(4, 4, dtype=torch.float64, generator=generator))
        naive.append(cmath.phase(complex(torch.linalg.qr(a)[0][0, 0])))
    assert concentration(naive) > 0.4


def test_random_unitary_heavy_output_probability():
    # A Haar-random state's heavy-output probability tends to (1 + ln 2) / 2.
    import math
    from blueqat.utils import random_unitary
    dim, trials = 64, 300
    total = 0.0
    for t in range(trials):
        probs = (random_unitary(dim, seed=5000 + t)[:, 0].abs() ** 2).tolist()
        ordered = sorted(probs)
        median = (ordered[dim // 2 - 1] + ordered[dim // 2]) / 2
        total += sum(p for p in probs if p > median)
    assert abs(total / trials - (1 + math.log(2)) / 2) < 0.02


def test_random_unitary_rejects_a_bad_dimension():
    import pytest
    from blueqat.utils import random_unitary
    with pytest.raises(ValueError):
        random_unitary(0)


# --- writing a QUBO down, and checking the answer --------------------------
#
# `from_qubo` existed and had no docstring, so a session that needed it wrote
# its own instead. The gap was discoverability, not capability.

def test_a_qubo_can_be_written_as_a_dictionary():
    """Which is how one is usually written down; the matrix form is awkward
    for anything sparse."""
    from blueqat import from_qubo
    matrix_form = from_qubo([[1, -2], [0, 3]])
    dict_form = from_qubo({(0, 0): 1, (1, 1): 3, (0, 1): -2})
    assert torch.allclose(matrix_form.to_matrix(), dict_form.to_matrix())


def test_the_two_triangles_are_summed():
    """So it does not matter which of (i, j) and (j, i) a caller uses."""
    from blueqat import from_qubo
    split = from_qubo({(0, 1): -1, (1, 0): -1})
    whole = from_qubo({(0, 1): -2})
    assert torch.allclose(split.to_matrix(), whole.to_matrix())


def test_a_malformed_qubo_key_says_what_a_key_looks_like():
    from blueqat import from_qubo
    with pytest.raises(TypeError, match='keyed by pairs'):
        from_qubo({0: 1.0})
    with pytest.raises(ValueError, match='non-negative'):
        from_qubo({(-1, 0): 1.0})


def test_the_exact_ground_state_energy():
    from blueqat import Z, ground_state_energy
    assert ground_state_energy(Z[0] * Z[1] + 0.5 * Z[0]) == pytest.approx(-1.5)
    assert ground_state_energy('Z[0]*Z[1] + 0.5*Z[0]') == pytest.approx(-1.5)


def test_the_ground_state_is_what_a_vqe_is_checked_against():
    """A converged variational answer means nothing until compared with one."""
    from blueqat import Z, ground_state_energy
    from blueqat.utils import QaoaAnsatz, Vqe
    hamiltonian = 1.0 * Z[0] * Z[1] - 0.5 * Z[1] * Z[2]
    exact = ground_state_energy(hamiltonian)
    result = Vqe(QaoaAnsatz(hamiltonian, step=3), seed=0).run(max_iter=200)
    assert float(result.loss_history[-1]) >= exact - 1e-9
    assert float(result.loss_history[-1]) == pytest.approx(exact, abs=0.2)


@pytest.mark.parametrize('qubo,value,bits', [
    ([[1, -2], [0, 3]], 0.0, (0, 0)),
    ({(0, 0): -1, (1, 1): -1, (0, 1): 2}, -1.0, (1, 0)),
    ({(0, 0): -1, (1, 1): -1}, -2.0, (1, 1)),
])
def test_the_exhaustive_minimum(qubo, value, bits):
    from blueqat import exhaustive_minimum
    assert exhaustive_minimum(qubo) == (pytest.approx(value), bits)


def test_a_tie_is_broken_the_same_way_every_time():
    """Arbitrary but fixed: a solver finding a different optimum of equal
    value has not disagreed with this one."""
    from blueqat import exhaustive_minimum
    value, bits = exhaustive_minimum({(0, 0): -1, (1, 1): -1, (0, 1): 2})
    assert value == pytest.approx(-1.0)
    assert bits == (1, 0)
    assert exhaustive_minimum({(0, 0): -1, (1, 1): -1, (0, 1): 2})[1] == bits


def test_they_are_exported_where_they_will_be_found():
    import blueqat
    for name in ('from_qubo', 'ground_state_energy', 'exhaustive_minimum'):
        assert hasattr(blueqat, name) and name in blueqat.__all__
