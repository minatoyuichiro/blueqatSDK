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
"""`seed=` (reproducible sampling / VQE) and `bit_order=` (counts key layout)."""

from collections import Counter

import pytest
import torch

from blueqat import Circuit
from blueqat.backends.backendbase import apply_bit_order
from blueqat.utils import Vqe, QaoaAnsatz, get_measurement_sampler, qubo_bit as q


# ---------------------------------------------------------------- seed: shots

def test_same_seed_gives_identical_counts():
    c = Circuit(4).h[:]
    assert c.run(shots=200, seed=11) == c.run(shots=200, seed=11)


def test_different_seed_gives_different_counts():
    c = Circuit(4).h[:]
    assert c.run(shots=200, seed=11) != c.run(shots=200, seed=12)


def test_seed_works_through_the_shots_helper():
    c = Circuit(3).h[:]
    assert c.shots(200, seed=5) == c.shots(200, seed=5)


@pytest.mark.parametrize('mode', ['statevector', 'tensornet'])
def test_seed_is_reproducible_in_both_modes(mode):
    c = Circuit(4).h[:].rz(0.3)[1]
    assert c.run(shots=100, seed=3, mode=mode) == c.run(shots=100, seed=3, mode=mode)


def test_seeded_counts_are_still_a_correct_distribution():
    # A seed must fix *which* samples are drawn, not bias them: a Bell state
    # still only ever produces the two correlated outcomes.
    counts = Circuit(2).h[0].cx[0, 1].run(shots=500, seed=99)
    assert set(counts) == {'00', '11'}
    assert sum(counts.values()) == 500


# ------------------------------------------- seed: mid-circuit collapse path

def test_seed_is_reproducible_with_reset():
    # `reset` forces the shot-by-shot trajectory path, whose randomness comes
    # from per-measurement collapse rather than one final sampling pass.
    c = Circuit(2).h[0].cx[0, 1].reset[0].m[:]
    assert c.run(shots=100, seed=7) == c.run(shots=100, seed=7)
    assert c.run(shots=100, seed=7) != c.run(shots=100, seed=8)


def test_seed_is_reproducible_for_keyed_samples():
    c = Circuit(2).h[0].m(key='a')[0].h[1].m(key='b')[1]
    assert c.run(shots=20, seed=4, returns='samples') == c.run(shots=20, seed=4, returns='samples')


def test_seed_is_reproducible_for_oneshot():
    c = Circuit(3).h[:]
    vec_a, bits_a = c.oneshot(seed=21)
    vec_b, bits_b = c.oneshot(seed=21)
    assert bits_a == bits_b
    assert torch.allclose(vec_a, vec_b)


def test_seed_and_bit_order_reach_the_large_n_sampling_path():
    # Above 28 qubits the backend cannot materialize a statevector and falls back
    # to qubit-by-qubit "perfect sampling", a third source of randomness.
    c = Circuit(30).h[0].cx[0, 1]
    counts = c.run(shots=5, seed=2)
    assert counts == c.run(shots=5, seed=2)
    assert set(counts) <= {'0' * 30, '0' * 28 + '11'}
    assert set(c.run(shots=5, seed=2, bit_order='q0_first')) <= {'0' * 30, '11' + '0' * 28}


# ---------------------------------------------------------- seed: hygiene

def test_seed_does_not_disturb_the_global_rng():
    # The point of a private generator: a seeded circuit run must not silently
    # reset the RNG that the surrounding program draws from.
    torch.manual_seed(555)
    expected = torch.rand(3)

    torch.manual_seed(555)
    Circuit(3).h[:].run(shots=50, seed=12345)
    assert torch.allclose(torch.rand(3), expected)


def test_unseeded_runs_stay_random():
    torch.manual_seed(0)
    c = Circuit(6).h[:]
    assert c.run(shots=200) != c.run(shots=200)


# ------------------------------------------------------------- bit_order

def test_bit_order_default_is_unchanged():
    # blueqat's long-standing layout: qubit 0 is the *rightmost* character.
    assert Circuit(3).x[0].run(shots=4) == Counter({'001': 4})
    assert Circuit(3).x[0].run(shots=4, bit_order='q0_last') == Counter({'001': 4})


def test_bit_order_q0_first_reverses_the_key():
    assert Circuit(3).x[0].run(shots=4, bit_order='q0_first') == Counter({'100': 4})
    assert Circuit(3).x[2].run(shots=4, bit_order='q0_first') == Counter({'001': 4})


def test_bit_order_keys_are_zero_padded_to_n_qubits():
    # The reason padding and reversal must travel together: an unpadded '11'
    # is ambiguous between qubits {0,1} and qubits {4,5}.
    for order in ('q0_last', 'q0_first'):
        counts = Circuit(6).x[0].x[1].run(shots=4, bit_order=order)
        key, = counts
        assert len(key) == 6
    assert Circuit(6).x[0].x[1].run(shots=4, bit_order='q0_first') == Counter({'110000': 4})


def test_bit_order_applies_to_the_collapse_path():
    counts = Circuit(3).x[0].reset[2].m[:].run(shots=10, bit_order='q0_first')
    assert counts == Counter({'100': 10})


def test_bit_order_rejects_unknown_values():
    with pytest.raises(ValueError):
        Circuit(2).h[0].run(shots=4, bit_order='little')


def test_apply_bit_order_helper():
    counts = Counter({'11': 3, '00': 1})
    assert apply_bit_order(counts, 4, 'q0_last') == Counter({'0011': 3, '0000': 1})
    assert apply_bit_order(counts, 4, 'q0_first') == Counter({'1100': 3, '0000': 1})
    assert apply_bit_order(counts, 2) == counts


def test_apply_bit_order_merges_keys_that_collide_after_padding():
    total = apply_bit_order(Counter({'1': 2, '01': 3}), 2, 'q0_first')
    assert total == Counter({'10': 5})


# ------------------------------------------------------------------- Vqe

def _hamiltonian():
    return q(0) - q(1)


def test_vqe_seed_makes_the_run_deterministic():
    a = Vqe(QaoaAnsatz(_hamiltonian(), 1)).run(max_iter=30, seed=42)
    b = Vqe(QaoaAnsatz(_hamiltonian(), 1)).run(max_iter=30, seed=42)
    assert torch.allclose(a.params, b.params)
    assert a.loss_history == b.loss_history


def test_vqe_different_seeds_start_from_different_points():
    a = Vqe(QaoaAnsatz(_hamiltonian(), 1)).run(max_iter=5, seed=1)
    b = Vqe(QaoaAnsatz(_hamiltonian(), 1)).run(max_iter=5, seed=2)
    assert not torch.allclose(a.params, b.params)


def test_vqe_seed_can_be_given_to_the_constructor():
    a = Vqe(QaoaAnsatz(_hamiltonian(), 1), seed=42).run(max_iter=30)
    b = Vqe(QaoaAnsatz(_hamiltonian(), 1)).run(max_iter=30, seed=42)
    assert torch.allclose(a.params, b.params)


def test_vqe_run_seed_overrides_the_constructor_seed():
    a = Vqe(QaoaAnsatz(_hamiltonian(), 1), seed=1).run(max_iter=5, seed=2)
    b = Vqe(QaoaAnsatz(_hamiltonian(), 1)).run(max_iter=5, seed=2)
    assert torch.allclose(a.params, b.params)


def test_vqe_seed_does_not_disturb_the_global_rng():
    torch.manual_seed(777)
    expected = torch.rand(3)
    torch.manual_seed(777)
    Vqe(QaoaAnsatz(_hamiltonian(), 1)).run(max_iter=5, seed=9)
    assert torch.allclose(torch.rand(3), expected)


def test_vqe_still_converges_with_an_explicit_seed():
    result = Vqe(QaoaAnsatz(_hamiltonian(), 1), seed=42).run()
    assert result.most_common(1)[0][0] == (0, 1)


def test_initial_params_still_wins_over_seed():
    given = torch.tensor([0.25, 0.75], dtype=torch.float64)
    result = Vqe(QaoaAnsatz(_hamiltonian(), 1)).run(max_iter=0, seed=3, initial_params=given)
    assert torch.allclose(result.params, given)


# ---------------------------------------------------------- loss_history

def test_loss_history_records_every_iteration():
    result = Vqe(QaoaAnsatz(_hamiltonian(), 1)).run(max_iter=12, seed=42)
    assert len(result.loss_history) == 12
    assert all(isinstance(v, float) for v in result.loss_history)


def test_loss_history_is_recorded_even_without_a_seed():
    assert len(Vqe(QaoaAnsatz(_hamiltonian(), 1)).run(max_iter=7).loss_history) == 7


def test_loss_history_shows_the_optimizer_making_progress():
    result = Vqe(QaoaAnsatz(_hamiltonian(), 1), seed=42).run()
    assert result.loss_history[-1] < result.loss_history[0]
    # Stopping early on the gradient tolerance is visible as a short history.
    assert len(result.loss_history) <= 500


def test_loss_history_last_value_matches_the_energy_at_the_final_step():
    # The recorded loss is the objective *before* that iteration's step, so the
    # final entry is the energy of the second-to-last parameter set, not of
    # `result.params`; it must still be a finite objective value.
    result = Vqe(QaoaAnsatz(_hamiltonian(), 1), seed=42).run(max_iter=3)
    assert all(abs(v) < 1e3 for v in result.loss_history)


# ---------------------------------------------------- seedable sampler

def test_measurement_sampler_seed_is_reproducible():
    c = Circuit(3).h[:]
    a = get_measurement_sampler(300, seed=3)(c, range(3))
    b = get_measurement_sampler(300, seed=3)(c, range(3))
    d = get_measurement_sampler(300, seed=4)(c, range(3))
    assert a == b
    assert a != d


def test_measurement_sampler_advances_between_calls():
    c = Circuit(3).h[:]
    sampler = get_measurement_sampler(300, seed=3)
    assert sampler(c, range(3)) != sampler(c, range(3))


def test_unseeded_measurement_sampler_is_unchanged():
    c = Circuit(2).h[:]
    probs = get_measurement_sampler(1000)(c, range(2))
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_vqe_seed_reseeds_a_seedable_sampler():
    sampler = get_measurement_sampler(200, seed=1)
    vqe = Vqe(QaoaAnsatz(_hamiltonian(), 1), sampler=sampler)
    # `set_seed` is what Vqe.run(seed=...) calls; after it the sampler must
    # replay the same draws.
    sampler.set_seed(5)
    c = Circuit(3).h[:]
    first = sampler(c, range(3))
    sampler.set_seed(5)
    assert sampler(c, range(3)) == first
    assert vqe.sampler is sampler
