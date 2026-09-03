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
"""Running a variational result on hardware, without any hardware."""

import json

import pytest
import torch

import blueqat.cloud as cloud
import blueqat.hardware as hardware
from blueqat import Circuit
from blueqat.utils import QaoaAnsatz, Vqe, X, Y, Z


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEQAT_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv(cloud.ENV_API_KEY, raising=False)
    cloud.reset_configuration()
    yield
    cloud.reset_configuration()


class FakeDevice:
    """A transport standing in for the hardware endpoints.

    Answers with the shapes measured on Toshiko: a submission returns a
    task_id, a status query returns `status` (or nothing at all while
    pending), and a result nests its counts under the classical register's
    name and states its own bit order.
    """

    def __init__(self, counts_for=None, status='COMPLETED', pending_first=0):
        self.counts_for = counts_for or (lambda payload: {"0" * payload["n_qubits"]: 256})
        self.status = status
        self.pending_first = pending_first
        self.submitted = []
        self.status_calls = 0

    def __call__(self, method, path, payload, api_key, endpoint):
        if method == 'POST' and path == '/hardware/jobs':
            self.submitted.append(payload)
            return {"task_id": f"task-{len(self.submitted)}", "status": "SUBMITTED",
                    "charged_jpy": "0"}
        if method == 'GET' and path.startswith('/hardware/jobs/') and path.endswith('/result'):
            index = int(path.split('/')[3].split('-')[1]) - 1
            return {"status": "COMPLETED",
                    "counts": {"c": self.counts_for(self.submitted[index])},
                    "bit_order": "bitstring[0] is qubit 0 (c[0])",
                    "cached": False}
        if method == 'GET' and path.startswith('/hardware/jobs/'):
            self.status_calls += 1
            if self.status_calls <= self.pending_first:
                return {}                      # measured: no status key while pending
            return {"status": self.status}
        raise AssertionError(f"unexpected call {method} {path}")


def _qaoa(hamiltonian, step=1, seed=0, max_iter=20):
    ansatz = QaoaAnsatz(hamiltonian, step=step)
    return ansatz, Vqe(ansatz, seed=seed).run(max_iter=max_iter)


# --- planning: what would this cost ----------------------------------------

def test_a_diagonal_hamiltonian_is_one_job_however_many_terms():
    """The point of the cache. QAOA's cost function is all Z, so every term's
    basis rotation is empty and every term asks for the same circuit. The
    service does no duplicate detection, so without this each term would be
    billed separately."""
    ansatz, result = _qaoa(1.0 * Z[0] * Z[1] + 0.5 * Z[1] * Z[2] + 0.3 * Z[0]
                           + 0.2 * Z[2] + 0.1 * Z[0] * Z[2])
    plan = hardware.HardwareEvaluation(ansatz, result.circuit).plan()
    assert plan["terms"] == 5
    assert plan["jobs"] == 1


def test_terms_in_different_bases_need_different_jobs():
    """X and Y terms rotate before measuring, so their circuits genuinely
    differ and cannot share a job."""
    ansatz, result = _qaoa(Z[0] * Z[1])
    ansatz.hamiltonian = (1.0 * Z[0] * Z[1] + 1.0 * X[0] * X[1]
                          + 1.0 * Y[0] * Y[1]).to_expr().simplify()
    plan = hardware.HardwareEvaluation(ansatz, result.circuit).plan()
    assert plan["jobs"] == 3


def test_the_plan_needs_no_key_and_submits_nothing():
    ansatz, result = _qaoa(Z[0] * Z[1])
    evaluation = hardware.HardwareEvaluation(ansatz, result.circuit, shots=1000)
    plan = evaluation.plan()          # no api key configured by the fixture
    assert plan["estimated_cost_jpy"] == pytest.approx(10.0 + 0.1 * 1000)
    assert plan["within_free_tier"] is False
    assert evaluation.task_ids == {}


def test_the_plan_counts_the_monthly_slots_it_would_use():
    ansatz, result = _qaoa(Z[0] * Z[1])
    assert hardware.HardwareEvaluation(ansatz, result.circuit).plan()["monthly_quota_used"] == 1


# --- refusing what the device would refuse ---------------------------------

def test_mid_circuit_measurement_is_refused_before_it_is_paid_for():
    """OQC rejects these at submission. A rejected job costs no money but does
    use one of the monthly allowance, so finding out remotely is not free."""
    ansatz, result = _qaoa(Z[0] * Z[1])
    broken = Circuit(2)
    broken.ops = list(result.circuit.ops)
    broken.m[0]
    broken.x[0]
    evaluation = hardware.HardwareEvaluation(ansatz, broken)
    with pytest.raises(ValueError, match='mid-circuit measurement'):
        evaluation.plan()


def test_too_many_qubits_is_refused_with_the_reason():
    ansatz, result = _qaoa(Z[0] * Z[1])
    wide = Circuit(40)
    wide.ops = list(result.circuit.ops)
    for q in range(40):
        wide.h[q]
    with pytest.raises(ValueError, match='allocates 40 qubits'):
        hardware.HardwareEvaluation(ansatz, wide, pack_qubits=False).plan()


def test_submitting_without_confirming_says_what_it_would_cost():
    ansatz, result = _qaoa(Z[0] * Z[1])
    evaluation = hardware.HardwareEvaluation(ansatz, result.circuit, shots=1000)
    cloud.configure(api_key='k', transport=FakeDevice())
    with pytest.raises(ValueError, match='costs money'):
        evaluation.submit()
    assert evaluation.task_ids == {}


# --- reading what comes back -----------------------------------------------

def test_counts_are_unwrapped_from_the_classical_register():
    """Hardware nests its counts under the register name; the simulator does
    not. Reading the simulator's shape off a hardware result is silent."""
    assert hardware.unwrap_counts({"c": {"01": 3}}) == {"01": 3}
    assert hardware.unwrap_counts({"01": 3}) == {"01": 3}
    with pytest.raises(ValueError, match='several classical registers'):
        hardware.unwrap_counts({"c": {"0": 1}, "d": {"1": 1}})


@pytest.mark.parametrize('described,expected', [
    ("bitstring[0] is qubit 0 (c[0])", "q0_first"),
    ("q0_last", "q0_last"),
    (None, "q0_first"),
])
def test_the_bit_order_is_read_from_the_result(described, expected):
    assert hardware.parse_bit_order(described) == expected


def test_an_unrecognised_bit_order_is_refused_rather_than_guessed():
    """Getting this backwards reports the mirror image with no error at all."""
    with pytest.raises(ValueError, match='Refusing to guess'):
        hardware.parse_bit_order("some new wording")


def test_marginals_respect_the_bit_order():
    counts = {"10": 100}                       # q0=1, q1=0 under q0_first
    first = hardware._marginal(counts, [0], 2, "q0_first")
    last = hardware._marginal(counts, [0], 2, "q0_last")
    assert first == {(1, ): 1.0}
    assert last == {(0, ): 1.0}


def test_a_result_wider_than_the_circuit_is_an_error():
    with pytest.raises(ValueError, match='not reporting what was asked for'):
        hardware._marginal({"1111": 10}, [0], 2)


# --- the whole chain, against an exactly known answer ----------------------

def _exact_counts(circuit, shots=1 << 20):
    """Counts with no sampling noise, in the device's own shape and order."""
    probs = (torch.abs(circuit.run()) ** 2).tolist()
    n = circuit.n_qubits
    out = {}
    for index, p in enumerate(probs):
        if p <= 0:
            continue
        bits = ''.join(str((index >> q) & 1) for q in range(n))   # q0 first
        out[bits] = out.get(bits, 0) + p * shots
    return {k: int(round(v)) for k, v in out.items() if round(v) > 0}


@pytest.mark.parametrize('hamiltonian', [
    1.0 * Z[0] * Z[1] + 0.5 * Z[1] + 0.25,
    -1.0 * Z[0] * Z[1] * Z[2] + 0.4 * Z[0] + 0.2 * Z[2],
])
def test_the_energy_matches_the_simulator_when_the_counts_are_exact(hamiltonian):
    """End to end: caching, packing, unwrapping, bit order and marginalization
    all have to be right together for this to land on the simulator's number."""
    ansatz, result = _qaoa(hamiltonian)
    evaluation = hardware.HardwareEvaluation(ansatz, result.circuit, shots=1 << 20)

    def counts_for(payload):
        circuit = Circuit(payload["n_qubits"])
        circuit.ops = list(evaluation._plan[0]["circuit"].ops)
        return _exact_counts(circuit)

    cloud.configure(api_key='k', transport=FakeDevice(counts_for=counts_for))
    evaluation.submit(confirm=True)
    evaluation.collect()

    exact = float(ansatz.get_energy_sparse(result.circuit))
    assert evaluation.energy() == pytest.approx(exact, abs=1e-6)


def test_probabilities_come_back_in_the_callers_numbering():
    ansatz, result = _qaoa(Z[0] * Z[1])

    def counts_for(payload):
        circuit = Circuit(payload["n_qubits"])
        circuit.ops = list(evaluation._plan[0]["circuit"].ops)
        return _exact_counts(circuit)

    evaluation = hardware.HardwareEvaluation(ansatz, result.circuit, shots=1 << 18)
    cloud.configure(api_key='k', transport=FakeDevice(counts_for=counts_for))
    evaluation.submit(confirm=True)
    probs = evaluation.probabilities()
    assert probs
    assert all(len(bits) == result.circuit.n_qubits for bits in probs)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)


# --- the uniform-noise correction ------------------------------------------

def test_removing_a_known_uniform_background_recovers_the_distribution():
    ideal = {(0, 0): 0.7, (0, 1): 0.3, (1, 0): 0.0, (1, 1): 0.0}
    f = 0.4
    noisy = {bits: f * p + (1 - f) / 4 for bits, p in ideal.items()}
    recovered = hardware.remove_uniform(noisy, 4, f)
    for bits in ideal:
        assert recovered[bits] == pytest.approx(ideal[bits], abs=1e-12)


def test_the_correction_divides_a_pauli_expectation_by_f():
    """<P> under the uniform distribution is zero for any non-identity Pauli,
    so a depolarized <P> is exactly f times the ideal one."""
    ideal = {(0, ): 0.8, (1, ): 0.2}
    f = 0.5
    noisy = {bits: f * p + (1 - f) / 2 for bits, p in ideal.items()}
    parity = lambda d: sum((-1) ** sum(b) * p for b, p in d.items())
    assert parity(noisy) == pytest.approx(f * parity(ideal))
    assert parity(hardware.remove_uniform(noisy, 2, f)) == pytest.approx(parity(ideal))


def test_there_is_no_estimator_for_the_noise_rate():
    """Deliberately: the obvious one reads the answer's own floor as noise
    unless the ideal distribution has an outcome of probability zero, which a
    VQE or QAOA circuit has no reason to have."""
    assert not hasattr(hardware, 'uniform_component')
    with pytest.raises(TypeError):
        hardware.remove_uniform({(0, ): 1.0}, 2)


@pytest.mark.parametrize('rate', [0.0, -0.1, 1.5])
def test_an_impossible_signal_fraction_is_refused(rate):
    with pytest.raises(ValueError, match='rate must be'):
        hardware.remove_uniform({(0, ): 1.0}, 2, rate)


# --- carrying a run across sessions ----------------------------------------

def test_a_submitted_run_can_be_collected_from_another_session():
    ansatz, result = _qaoa(Z[0] * Z[1])
    evaluation = hardware.HardwareEvaluation(ansatz, result.circuit)
    device = FakeDevice()
    cloud.configure(api_key='k', transport=device)
    evaluation.submit(confirm=True)
    carried = json.loads(json.dumps(evaluation.to_dict()))

    revived = hardware.HardwareEvaluation.from_dict(ansatz, result.circuit, carried)
    assert revived.task_ids == evaluation.task_ids
    revived.collect()
    assert revived.counts


def test_collecting_against_a_different_circuit_is_refused():
    """Otherwise the stored measurements would be attached to the wrong terms,
    and the energy would come out wrong with no error."""
    ansatz, result = _qaoa(Z[0] * Z[1])
    evaluation = hardware.HardwareEvaluation(ansatz, result.circuit)
    cloud.configure(api_key='k', transport=FakeDevice())
    evaluation.submit(confirm=True)

    other = Circuit(2).h[0].h[1]
    with pytest.raises(ValueError, match='do not match'):
        hardware.HardwareEvaluation.from_dict(ansatz, other, evaluation.to_dict())


# --- status and failure ----------------------------------------------------

def test_a_missing_status_field_reads_as_pending_rather_than_crashing():
    """Measured: a pending job answers with no status key at all."""
    ansatz, result = _qaoa(Z[0] * Z[1])
    evaluation = hardware.HardwareEvaluation(ansatz, result.circuit)
    # Two pending answers: one for the status() below and one for ready().
    cloud.configure(api_key='k', transport=FakeDevice(pending_first=2))
    evaluation.submit(confirm=True)
    assert set(evaluation.status().values()) == {'pending'}
    assert evaluation.ready() is False


def test_a_failure_quotes_the_message_and_distrusts_the_code():
    """error_code 101 was measured covering three unrelated causes, so the
    message is the only thing that identifies what went wrong."""
    ansatz, result = _qaoa(Z[0] * Z[1])
    evaluation = hardware.HardwareEvaluation(ansatz, result.circuit)

    class Failing(FakeDevice):
        def __call__(self, method, path, payload, api_key, endpoint):
            if method == 'GET' and path.startswith('/hardware/jobs/'):
                return {"status": "FAILED", "error_code": 101,
                        "error_message": "Attempted to allocate more qubits than available."}
            return super().__call__(method, path, payload, api_key, endpoint)

    cloud.configure(api_key='k', transport=Failing())
    evaluation.submit(confirm=True)
    with pytest.raises(RuntimeError) as exc:
        evaluation.collect()
    message = str(exc.value)
    assert 'Attempted to allocate more qubits' in message
    assert 'not specific' in message
    assert "account's history" in message


def test_collecting_before_a_job_finishes_says_so():
    ansatz, result = _qaoa(Z[0] * Z[1])
    evaluation = hardware.HardwareEvaluation(ansatz, result.circuit)
    cloud.configure(api_key='k', transport=FakeDevice(status='SUBMITTED'))
    evaluation.submit(confirm=True)
    with pytest.raises(RuntimeError, match='not finished'):
        evaluation.collect()


def test_wait_returns_as_soon_as_the_jobs_are_done(monkeypatch):
    """Jobs were measured completing in 15 to 25 seconds, so the poll starts
    fine-grained; this only checks it does not sleep when there is nothing to
    wait for."""
    slept = []
    monkeypatch.setattr(hardware.time, 'sleep', lambda s: slept.append(s))
    ansatz, result = _qaoa(Z[0] * Z[1])
    evaluation = hardware.HardwareEvaluation(ansatz, result.circuit)
    cloud.configure(api_key='k', transport=FakeDevice())
    evaluation.submit(confirm=True)
    evaluation.wait()
    assert slept == []


def test_wait_gives_up_without_losing_the_task_ids(monkeypatch):
    monkeypatch.setattr(hardware.time, 'sleep', lambda s: None)
    ansatz, result = _qaoa(Z[0] * Z[1])
    evaluation = hardware.HardwareEvaluation(ansatz, result.circuit)
    cloud.configure(api_key='k', transport=FakeDevice(status='SUBMITTED'))
    evaluation.submit(confirm=True)
    with pytest.raises(TimeoutError, match='task_ids'):
        evaluation.wait(timeout=0.0)
    assert evaluation.task_ids


# --- packing the register --------------------------------------------------

def test_a_gappy_register_is_packed_and_reported_back_in_the_original_numbering():
    circuit = Circuit(20).h[13].cx[13, 18]
    packed, back = hardware.compact(circuit)
    assert packed.n_qubits == 2
    assert back == {0: 13, 1: 18}


def test_packing_leaves_an_already_compact_circuit_alone():
    circuit = Circuit(3).h[0].cx[0, 1].cx[1, 2]
    packed, back = hardware.compact(circuit)
    assert packed is circuit
    assert back == {0: 0, 1: 1, 2: 2}


# --- is this correction even the right one ---------------------------------
#
# `q = f q_ideal + (1-f)/N` is a model, and it has been seen not to hold on the
# device: analysing real Toshiko output, what survived was a product
# distribution -- right single-qubit marginals, no correlation -- rather than a
# uniform background. Dividing out a uniform component cannot restore a
# correlation that is gone, so the shape is worth reporting.

def _product(marginals):
    import itertools
    out = {}
    for bits in itertools.product((0, 1), repeat=len(marginals)):
        weight = 1.0
        for m, b in zip(marginals, bits):
            weight *= m if b else 1.0 - m
        out[bits] = weight
    return out


def test_uniform_noise_leaves_the_correlations_visible():
    ghz = {(0, 0, 0): 0.5, (1, 1, 1): 0.5}
    f = 0.4
    diluted = {bits: f * ghz.get(bits, 0.0) + (1 - f) / 8
               for bits in _product([0.5, 0.5, 0.5])}
    shape = hardware.noise_shape(diluted)
    assert shape["to_product"] > 0.2      # correlations survive: correcting helps


def test_a_product_residual_is_flagged_even_though_it_is_not_uniform():
    """The measured case: marginals pulled toward 0.5 but not at it, which a
    uniform background cannot produce."""
    shape = hardware.noise_shape(_product([0.37, 0.47, 0.47]))
    assert shape["to_product"] == pytest.approx(0.0, abs=1e-12)
    assert shape["to_uniform"] > 0.1      # not uniform, yet uncorrelated


def test_the_two_measures_agree_on_a_genuinely_uniform_distribution():
    shape = hardware.noise_shape(_product([0.5, 0.5, 0.5]))
    assert shape["to_uniform"] == pytest.approx(0.0, abs=1e-12)
    assert shape["to_product"] == pytest.approx(0.0, abs=1e-12)


def test_total_variation_is_symmetric_and_zero_on_itself():
    a = {(0, ): 0.3, (1, ): 0.7}
    b = {(0, ): 0.5, (1, ): 0.5}
    assert hardware.total_variation(a, a) == pytest.approx(0.0)
    assert hardware.total_variation(a, b) == pytest.approx(hardware.total_variation(b, a))
    assert hardware.total_variation(a, b) == pytest.approx(0.2)


# --- the pre-submit hook ---------------------------------------------------

def test_a_hook_can_stop_the_submission():
    """Kept a hook rather than a built-in check so the checker stays a
    dependency of the caller."""
    ansatz, result = _qaoa(Z[0] * Z[1])
    seen = {}

    def veto(evaluation, summary):
        seen.update(summary)
        raise RuntimeError("this formulation is degenerate")

    evaluation = hardware.HardwareEvaluation(ansatz, result.circuit, before_submit=veto)
    device = FakeDevice()
    cloud.configure(api_key='k', transport=device)
    with pytest.raises(RuntimeError, match='degenerate'):
        evaluation.submit(confirm=True)
    assert device.submitted == []          # nothing was sent
    assert seen["jobs"] == 1               # and it saw the plan


def test_a_hook_that_returns_lets_the_submission_through():
    ansatz, result = _qaoa(Z[0] * Z[1])
    calls = []
    evaluation = hardware.HardwareEvaluation(
        ansatz, result.circuit,
        before_submit=lambda ev, summary: calls.append(summary))
    cloud.configure(api_key='k', transport=FakeDevice())
    evaluation.submit(confirm=True)
    assert len(calls) == 1
    assert evaluation.task_ids
