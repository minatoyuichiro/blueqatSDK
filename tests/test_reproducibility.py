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


import warnings
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


# --- which end of a counts key is qubit 0 ----------------------------------
#
# blueqat 2.0.4's numpy and numba backends put qubit 0 at the *left*, giving
# '100' where this gives '001' for the same circuit -- the mirror image, with
# no error and no warning. Both versions call themselves blueqat, so code that
# reads a bitstring is version-dependent in a way nothing announces. The
# convention is exported so a caller can assert it and fail loudly on a version
# that would answer backwards.

def test_qubit_zero_is_the_last_character():
    import blueqat
    circuit = Circuit(3)
    circuit.x[0]
    assert dict(circuit.m[:].run(shots=1)) == {'001': 1}
    assert blueqat.BIT_ORDER == 'q0_last'


def test_every_backend_agrees_on_the_bit_order():
    """Choosing a backend must not change which end is which -- that is exactly
    the failure being guarded against."""
    import blueqat
    for backend in ('statevector', 'tensornet'):
        circuit = Circuit(3)
        circuit.x[0]
        assert dict(circuit.m[:].run(backend=backend, shots=1)) == {'001': 1}, backend
    assert blueqat.BIT_ORDER == 'q0_last'


def test_the_convention_is_exported_so_a_guard_is_one_line():
    import blueqat
    assert 'BIT_ORDER' in blueqat.__all__
    assert blueqat.BIT_ORDER in ('q0_last', 'q0_first')


def test_the_measured_order_agrees_with_the_declared_one():
    """If these ever disagree, the constant is lying and every guard built on
    it is wrong."""
    import blueqat
    assert blueqat.measure_bit_order() == blueqat.BIT_ORDER


def test_measuring_works_where_the_constant_would_not_exist():
    """The constant was added after 2.1.3 shipped, so an installed 2.1.3 --
    which answers q0_last, correctly -- has no attribute to read. Treating its
    absence as "old and wrong" would reject a working install; measuring is
    right about whatever is actually there."""
    import blueqat
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.delattr(blueqat, 'BIT_ORDER')
    try:
        order = getattr(blueqat, 'BIT_ORDER', None) or blueqat.measure_bit_order()
        assert order == 'q0_last'
    finally:
        monkeypatch.undo()
    assert blueqat.BIT_ORDER == 'q0_last'


def test_the_documented_two_liner_gives_the_same_answer():
    """Code that cannot import measure_bit_order copies these two lines, so
    they have to stay correct."""
    import blueqat
    circuit = Circuit(3)
    circuit.x[0]
    order = "q0_last" if "001" in circuit.m[:].run(shots=1) else "q0_first"
    assert order == blueqat.BIT_ORDER


# --- asking for an expectation value ---------------------------------------

def test_a_hamiltonian_that_is_not_one_says_what_one_looks_like():
    """The old failure was `AttributeError: 'dict' object has no attribute
    'simplify'` from three frames down, which says nothing about
    Hamiltonians."""
    with pytest.raises(TypeError) as exc:
        Circuit(1).expect({'Z0': 1.0})
    message = str(exc.value)
    assert 'Pauli expression' in message
    assert 'from blueqat import Z' in message


def test_a_hamiltonian_can_be_written_as_a_string():
    from blueqat import Z
    assert Circuit(2).x[0].expect('Z[0] + 0.5*Z[1]') == pytest.approx(-0.5)
    assert Circuit(2).x[0].expect(Z[0]) == pytest.approx(-1.0)


def test_the_do_nothing_state_has_an_expectation_value():
    """It is the first thing anyone measures, and it used to raise about a
    zero-qubit state. Qubits the circuit never mentions are in |0>."""
    from blueqat import Z
    assert Circuit().expect(Z[0]) == pytest.approx(1.0)
    assert Circuit(1).h[0].expect(Z[1]) == pytest.approx(1.0)


def test_asking_does_not_widen_the_circuit_being_asked_about():
    from blueqat import Z
    circuit = Circuit(1).h[0]
    circuit.expect(Z[3])
    assert circuit.n_qubits == 1


def test_the_pauli_operators_are_exported_at_the_top_level():
    """They are how a Hamiltonian is written, so they belong on the surface
    rather than only in a module named for miscellany."""
    import blueqat
    for name in ('I', 'X', 'Y', 'Z', 'Expr', 'Term', 'parse_hamiltonian'):
        assert hasattr(blueqat, name), name
        assert name in blueqat.__all__, name
    from blueqat import X, Z
    assert Circuit(2).x[0].expect(Z[0] + 0.5 * X[1]) == pytest.approx(-1.0)


# --- probs() and run(shots=) agree about which end is which ----------------

def test_probs_takes_the_same_bit_order_argument_as_run():
    """One of them having the argument and the other not is how a reader ends
    up believing they differ."""
    circuit = Circuit(3).x[0]
    default = circuit.probs()
    assert int(default.argmax()) == 1                  # index 1 == 001
    reversed_ = circuit.probs(bit_order='q0_first')
    assert int(reversed_.argmax()) == 4                # index 4 == 100
    assert float(reversed_.sum()) == pytest.approx(1.0)

    counts = Circuit(3).x[0].m[:].run(shots=1)
    assert list(counts) == ['001']
    counts = Circuit(3).x[0].m[:].run(shots=1, bit_order='q0_first')
    assert list(counts) == ['100']


def test_the_bit_order_applies_to_a_marginal_too():
    circuit = Circuit(3).x[0]
    assert circuit.probs([0, 1]).tolist() == [0.0, 1.0, 0.0, 0.0]
    assert circuit.probs([0, 1], bit_order='q0_first').tolist() == [0.0, 0.0, 1.0, 0.0]


def test_an_unknown_bit_order_is_refused():
    with pytest.raises(ValueError, match="q0_last.*q0_first"):
        Circuit(2).h[0].probs(bit_order='big_endian')


def test_reversing_is_the_identity_on_a_symmetric_state():
    """Which is precisely why reading the convention backwards survives a
    whole set of examples and then fails on the one that is not symmetric."""
    ghz = Circuit(3).h[0].cx[0, 1].cx[1, 2]
    assert torch.allclose(ghz.probs(), ghz.probs(bit_order='q0_first'))
    asymmetric = Circuit(3).x[0]
    assert not torch.allclose(asymmetric.probs(),
                              asymmetric.probs(bit_order='q0_first'))


def test_the_statevector_says_which_end_is_qubit_zero():
    """It said nothing at all, in 83 characters."""
    doc = Circuit.statevector.__doc__
    assert 'least' in doc and 'significant' in doc
    state = Circuit(2).x[0].run()
    assert int(torch.abs(state).argmax()) == 1        # |01>, qubit 0 set


def test_statevector_honours_bit_order_rather_than_accepting_and_ignoring_it():
    """Worse than absent, before this: the argument was accepted and even
    validated, then silently ignored, so asking for the other convention
    returned the default one with nothing said."""
    circuit = Circuit(3).x[0]
    default = circuit.statevector()
    reversed_ = circuit.statevector(bit_order='q0_first')
    assert int(torch.abs(default).argmax()) == 1        # 001
    assert int(torch.abs(reversed_).argmax()) == 4      # 100
    assert not torch.allclose(default, reversed_)
    assert float((torch.abs(reversed_) ** 2).sum()) == pytest.approx(1.0)


def test_all_three_ways_of_reading_a_state_take_the_same_argument():
    """Having it on some and not the others is the trap: "I checked with
    run()" then stops being an answer about the other two."""
    import inspect
    for name in ('statevector', 'probs'):
        assert 'bit_order' in inspect.signature(getattr(Circuit, name)).parameters
        with pytest.raises(ValueError, match='bit_order must be one of'):
            getattr(Circuit(2).h[0], name)(bit_order='big_endian')
    # run() takes it as a keyword, and all three agree on the same circuit.
    assert list(Circuit(3).x[0].m[:].run(shots=1, bit_order='q0_first')) == ['100']
    assert int(Circuit(3).x[0].probs(bit_order='q0_first').argmax()) == 4
    assert int(torch.abs(Circuit(3).x[0].statevector(bit_order='q0_first')).argmax()) == 4


def test_reversing_the_statevector_keeps_the_gradient():
    angle = torch.tensor(0.3, requires_grad=True)
    state = Circuit(2).ry(angle)[0].statevector(bit_order='q0_first')
    (torch.abs(state) ** 2).sum().backward()
    assert angle.grad is not None


# --- a gate written without its qubits -------------------------------------
#
# `circuit.ry(theta, 0)` is how several other toolkits are written. blueqat
# spells the qubit as a subscript, so that call built a gate and threw it
# away: no error, no warning, and a circuit quietly missing it. A model
# working a benchmark hit this, saw `Circuit().ry(1.0, 0).run()` return
# `[1.+0.j]`, concluded that blueqat does not apply RY, and rewrote its whole
# ansatz in numpy. It was not wrong about what it saw.

@pytest.mark.parametrize('call,suggestion', [
    ('c.ry(1.0, 0)', 'ry(1.0)[0]'),
    ('c.rx(0.5, 2)', 'rx(0.5)[2]'),
    ('c.h(0)', 'h[0]'),
    ('c.x(0)', 'x[0]'),
    ('c.cx(0, 1)', 'cx[0, 1]'),
    ('c.ccx(0, 1, 2)', 'ccx[0, 1, 2]'),
])
def test_a_gate_without_a_subscript_says_so_and_spells_the_fix(call, suggestion):
    import gc
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        c = Circuit()
        eval(call)
        del c
        gc.collect()
    messages = [str(w.message) for w in caught if issubclass(w.category, SyntaxWarning)]
    assert len(messages) == 1
    assert suggestion in messages[0]
    assert 'no gate was added' in messages[0]


def test_correctly_written_gates_say_nothing():
    import gc
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        c = Circuit(2).ry(1.0)[0].h[1].cx[0, 1].m[:]
        del c
        gc.collect()
    assert [w for w in caught if issubclass(w.category, SyntaxWarning)] == []


def test_an_empty_circuit_is_not_a_mistake():
    """Deliberately not warned about. Running a circuit with no gates is a
    real question -- it is the |0...0> state, and `Circuit().expect(Z[0])`
    answering 1 was itself a fix. Warning here would fire where nothing is
    wrong, and the discarded-gate warning already catches the actual error."""
    import gc
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        state = Circuit(3).run()
        del state
        gc.collect()
    assert [w for w in caught if issubclass(w.category, SyntaxWarning)] == []


def test_the_other_toolkit_spelling_is_refused_when_it_is_subscripted():
    """`ry(theta, qubit)[q]` used to work by silently dropping the qubit
    argument, which is the same mistake surviving into a circuit that looks
    right."""
    with pytest.raises(ValueError) as exc:
        Circuit(1).ry(1.0, 0)[0]
    message = str(exc.value)
    assert 'takes one parameter' in message
    assert 'subscript, not an argument' in message
    assert 'ry(1.0)[0]' in message


def test_a_rotation_with_no_angle_says_which_is_missing():
    with pytest.raises(ValueError, match='takes one parameter, and none was given'):
        Circuit(1).ry()[0]


def test_the_benchmark_sequence_now_reports_itself():
    """The exact three lines that led to the misdiagnosis."""
    import gc
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        c3 = Circuit()
        c3.ry(1.0, 0)
        state = c3.run()
        gc.collect()
    assert state.shape == (1, )            # still a zero-qubit circuit
    assert any('no gate was added' in str(w.message) for w in caught)


# --- an argument that was accepted and will not be used --------------------
#
# The same disease as a gate written without its qubits, one level up. A
# misspelled keyword is accepted and dropped, so `run(shot=100)` returns a
# statevector when counts were asked for, `run(seeed=1)` is not seeded and
# looks seeded, and `run(bit_oder=...)` quietly gives the other bit order --
# the mistake that has already cost one benchmark answer.

@pytest.mark.parametrize('bad,good', [
    ('shot', 'shots'),
    ('seeed', 'seed'),
    ('bit_oder', 'bit_order'),
    ('hamiltonain', 'hamiltonian'),
])
def test_a_misspelled_argument_is_named_with_its_correction(bad, good):
    circuit = Circuit(2).h[0].cx[0, 1].m[:]
    with pytest.warns(UserWarning) as caught:
        circuit.run(**{bad: 1}, shots=4, seed=1)
    message = str(caught[0].message)
    assert f"'{bad}'" in message
    assert f"did you mean '{good}'" in message
    assert 'does not use' in message


def test_correct_arguments_say_nothing():
    circuit = Circuit(2).h[0].cx[0, 1].m[:]
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        circuit.run(shots=4, seed=1, bit_order='q0_first')
        circuit.run()
        circuit.run(backend='density', shots=4)


def test_an_argument_with_no_near_miss_is_still_reported():
    """It is not only typos: an argument meant for another backend is dropped
    just as quietly."""
    with pytest.warns(UserWarning, match='does not use'):
        Circuit(2).h[0].run(wildly_unrelated=True)


def test_the_warning_says_what_the_backend_does_accept():
    with pytest.warns(UserWarning) as caught:
        Circuit(2).h[0].run(shot=4)
    message = str(caught[0].message)
    for name in ('shots', 'seed', 'bit_order'):
        assert name in message


def test_circuit_to_unitary_no_longer_passes_an_argument_nobody_reads():
    """`ignore_global` was defaulted into every call and read by nothing --
    `ignore_global_phase` is a separate utility. It cost nothing and meant
    nothing, which is how it survived; the check above found it by warning
    1320 times in one test run."""
    from blueqat.circuit_funcs import circuit_to_unitary
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        matrix = circuit_to_unitary(Circuit(2).h[0].cx[0, 1])
    assert matrix.shape == (4, 4)
