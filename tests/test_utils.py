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
