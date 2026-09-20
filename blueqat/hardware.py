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
"""Running a variational result on real hardware.

The usual shape of a VQE or QAOA run is: optimize against a simulator, then
evaluate the answer once on a device. `HardwareEvaluation` is that second step.

It is in two phases, because hardware is not a function call:

- `plan()` works out exactly which circuits the evaluation needs and what they
  would cost. Nothing is submitted and no API key is required.
- `submit(confirm=True)` sends them; `collect()` or `wait()` brings the counts
  back, and `energy()` turns them into a number.

Submitting and collecting are separate so that the run can outlive the session
that started it. Measured jobs have come back in 15 to 25 seconds, but the
device is not always accepting work -- OQC's Toshiko opens twice a day on
weekdays -- and a submission left over a weekend has been measured taking 32 to
56 hours. `task_ids` and `to_dict`/`from_dict` carry a run across that.

The plan is not derived by reimplementing which circuits an evaluation needs.
It is obtained by running the evaluation against a sampler that records what it
is asked for, so the enumeration cannot drift away from what the real
evaluation does.

Numbers quoted in this module come from measurements by the `tnapi` and
`blueqatmcp` sessions on Toshiko itself (2026-09-01), recorded in
``~/pm/QUANTUM_HW.md``.
"""

import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .circuit import Circuit
from .gate import Measurement

#: What the service charges, from ``~/pm/QUANTUM_HW.md``. Used only to show a
#: number before spending it; the authority is `cloud.hardware_quote`.
JOB_FEE_JPY = 10.0
SHOT_FEE_JPY = 0.1
FREE_TIER_SHOTS = 256

#: Toshiko exposes 32 qubits. Its own numbering runs 1..35 with gaps, but
#: `n_qubits` on submission means *how many to allocate*, not the highest index
#: plus one -- asking for 36 was measured failing with "Attempted to allocate
#: more qubits than available."
MAX_HARDWARE_QUBITS = 32

#: The three states actually observed on Toshiko. A job's status field can also
#: be missing entirely while it is pending, so read it defensively.
DONE_STATES = frozenset({'completed', 'done', 'succeeded', 'success', 'finished'})
FAILED_STATES = frozenset({'failed', 'cancelled', 'canceled', 'error'})


class _Recorder:
    """A sampler that records what it is asked for and answers with nonsense.

    The answers never reach a result: `plan()` throws away the value it
    computes and keeps only the record. Returning a uniform distribution rather
    than raising keeps the caller's arithmetic well-defined while it walks
    every term.
    """

    def __init__(self) -> None:
        self.requests: List[Tuple[Circuit, Tuple[int, ...]]] = []

    def __call__(self, circuit: Circuit, meas) -> Dict[Tuple[int, ...], float]:
        meas = tuple(int(q) for q in meas)
        self.requests.append((circuit, meas))
        size = 1 << len(meas)
        weight = 1.0 / size
        return {tuple((index >> k) & 1 for k in range(len(meas))): weight
                for index in range(size)}


def circuit_key(circuit: Circuit) -> str:
    """A stable identity for a circuit, as the wire format sees it.

    Two circuits with the same key produce the same job, so they are submitted
    once. That is not a micro-optimization. `AnsatzBase.get_energy` calls its
    sampler once per Hamiltonian term, and for a diagonal Hamiltonian -- every
    QAOA cost function -- the basis rotations are empty, so every one of those
    calls asks for the *same* circuit. The service does no duplicate detection
    of its own (one call is one task), so without this a QAOA evaluation would
    pay for one job per term instead of one job.
    """
    from .cloud import circuit_to_gates
    n, gates = circuit_to_gates(circuit)
    return json.dumps({"n": n, "gates": gates}, sort_keys=True)


def used_qubits(circuit: Circuit) -> List[int]:
    """The qubit indices a circuit actually touches, in order."""
    from .circuit_funcs.flatten import flatten
    touched = set()
    for op in flatten(circuit).ops:
        touched.update(int(t) for t in op.target_iter(circuit.n_qubits))
    return sorted(touched)


def compact(circuit: Circuit) -> Tuple[Circuit, Dict[int, int]]:
    """Renumber a circuit's qubits into ``0..k-1``, keeping their order.

    This also shrinks the register, which is the part with a clear reason
    behind it: `n_qubits` at submission is how many qubits to *allocate*, and
    Toshiko has 32, so a circuit written on high indices can be refused outright
    for asking for too many.

    Accuracy may improve too, but the evidence for that is weaker than it looks
    and is quoted here so nobody leans on it: the same circuit moved from qubits
    13/15/18 to 0/1/2/3 went from a total variation of 0.5463 to 0.3637 -- but
    that measurement changed the register width (19 to 4) at the same time as
    the numbering, so which of the two mattered is not separated. It is on
    `tnapi`'s list to redo one variable at a time.

    The returned map sends each new index back to the original, so counts can
    be reported in the caller's numbering.
    """
    from .circuit_funcs.flatten import flatten
    original = used_qubits(circuit)
    if original == list(range(len(original))) and circuit.n_qubits == len(original):
        return circuit, {q: q for q in original}
    forward = {old: new for new, old in enumerate(original)}
    packed = Circuit(len(original))
    for op in flatten(circuit).ops:
        packed.ops.append(_retarget(
            op, tuple(forward[int(t)] for t in op.target_iter(circuit.n_qubits))))
    return packed, {new: old for old, new in forward.items()}


def _retarget(op, targets: Tuple[int, ...]):
    """A copy of `op` acting on `targets`."""
    import copy
    moved = copy.copy(op)
    moved.targets = targets
    return moved


def unwrap_counts(counts: Any) -> Dict[str, int]:
    """Counts as a flat ``{bitstring: n}``, whatever the service wrapped them in.

    The simulator and the device do not agree on this, and assuming the
    simulator's shape is how a hardware run silently reads the wrong bits.
    ``/circuits/run`` normalizes its counts; a hardware result passes through
    what OQC returned, which is nested under the classical register's name:
    ``{"counts": {"c": {"0000": 47, ...}}}``.
    """
    if isinstance(counts, dict) and counts and all(
            isinstance(v, dict) for v in counts.values()):
        if len(counts) > 1:
            raise ValueError(
                f"counts came back split across several classical registers "
                f"({sorted(counts)}); which one holds the measurement is not "
                f"something this can guess.")
        counts = next(iter(counts.values()))
    if not isinstance(counts, dict):
        raise ValueError(f"cannot read counts of type {type(counts).__name__}.")
    return {str(k): int(v) for k, v in counts.items()}


def parse_bit_order(described: Optional[str]) -> str:
    """Read the service's own description of its bit order.

    Hardware results carry a `bit_order` field, measured reading
    ``"bitstring[0] is qubit 0 (c[0])"`` -- the opposite of blueqat's own
    convention. Taking it from the result rather than hardcoding either one
    means a change on the far side shows up as an error instead of as a
    mirrored answer.
    """
    if not described:
        return "q0_first"          # what Toshiko was measured returning
    text = str(described).lower()
    if 'bitstring[0] is qubit 0' in text or 'q0_first' in text:
        return "q0_first"
    if 'bitstring[-1] is qubit 0' in text or 'q0_last' in text:
        return "q0_last"
    raise ValueError(
        f"the service described its bit order as {described!r}, which this does "
        f"not recognise. Refusing to guess: getting it backwards reports the "
        f"mirror image of the answer without any error.")


def _marginal(counts: Dict[str, int], meas: Sequence[int], n_qubits: int,
              bit_order: str = "q0_first") -> Dict[Tuple[int, ...], float]:
    """Probabilities over `meas` only, summing out every other qubit.

    `bit_order` says which end of a key is qubit 0. A device's convention is
    its own, and getting it backwards silently reports the mirror image, so it
    is a parameter rather than an assumption. Keys were measured coming back at
    exactly the submitted width, with unused qubits reading zero.
    """
    if bit_order not in ("q0_last", "q0_first"):
        raise ValueError(f"bit_order must be 'q0_last' or 'q0_first', got {bit_order!r}.")
    total = sum(counts.values())
    if not total:
        return {}
    out: Dict[Tuple[int, ...], float] = {}
    for key, n in counts.items():
        if len(key) > n_qubits:
            raise ValueError(
                f"a result key is {len(key)} characters wide but the circuit has "
                f"{n_qubits} qubits; the device is not reporting what was asked for.")
        if bit_order == "q0_first":
            padded = key.ljust(n_qubits, '0')
            bits = tuple(int(padded[q]) for q in meas)
        else:
            padded = key.zfill(n_qubits)
            bits = tuple(int(padded[n_qubits - 1 - q]) for q in meas)
        out[bits] = out.get(bits, 0.0) + n / total
    return out


def remove_uniform(probs: Dict[Tuple[int, ...], float], n_outcomes: int,
                   rate: float) -> Dict[Tuple[int, ...], float]:
    """Undo a uniform (depolarizing) background of weight `rate`.

    A device's output is described well by ``q = (1-f) * uniform + f *
    q_ideal``. On Toshiko, removing the uniform part of a demo circuit's output
    took its total variation from the exact distribution from 0.3378 to 0.1132,
    and the most likely outcome from 0.305 to 0.573 (exact 0.625) -- one
    subtraction, no extra hardware job.

    On a Pauli expectation this is exactly a division by `f`: a non-identity
    Pauli averages to zero over the uniform distribution, so ``<P>_measured ==
    f * <P>_ideal``. Doing it on the distribution keeps that consistent with
    the sampled probabilities, and leaves the Hamiltonian's constant term alone
    -- which is correct and easy to get wrong, since the identity term is *not*
    suppressed by depolarizing noise and must not be divided. `get_energy` adds
    it without consulting the sampler, so it never passes through here.

    ⚠ Before using this at all, check `noise_shape`. The model behind it --
    ``q = f q_ideal + (1-f)/N`` -- has been seen not to hold on the device:
    analysing real Toshiko output, what survived was a *product* distribution
    (marginals 0.37/0.47/0.47, within 0.0395 of the product of its own
    marginals) rather than a uniform background, and no amount of dividing
    restores a correlation that is gone.

    ⚠ `rate` has to be measured; there is deliberately no estimator here.
    The obvious one -- read the background off the smallest observed
    probability -- only works when the ideal distribution has an outcome of
    probability zero, because ``N * min(q) == (1-f) + N * f * min(q_ideal)``.
    That held for the run those numbers come from, which encoded 5 states in 8
    and could read `f` off the 3 that should have been empty. A VQE or QAOA
    circuit has no forbidden bitstrings, so it offers no such foothold, and an
    estimate taken this way would read the answer's own floor as noise and
    inflate the result.

    The ways of measuring `f` all have a catch, and none is done automatically
    here:

    - A **mirror circuit** (run ``U`` then ``U†``, which should return all
      zeros) has the strongest argument on paper: same gates, same layout, and
      under global depolarizing noise ``f_mirror == f**2`` exactly -- verified
      in density-matrix simulation, with a 1 to 4 percent overestimate under
      local depolarizing noise. ⚠ On the device that extrapolation does not
      survive: measured Bell-plus-CX fidelities of 0.789, 0.711, 0.742 and
      0.539 at 1, 3, 5 and 9 CX are not monotonic and give per-CX ratios of
      0.949, 0.985 and 0.954. Squaring and square-rooting has no measured basis
      on this hardware. A mirror also doubles the depth against a CX budget of
      roughly 15.
    - A **reference circuit** with a known answer costs an extra job, and
      cannot be placed on the same physical qubits: `preserve_layout` is
      refused by Toshiko, and the compiler's node numbering does not match the
      calibration's. With per-pair CX fidelity ranging from 0.9537 to 0.9859,
      the same depth elsewhere on the chip is not the same `f`. It must also
      match the number of qubits *measured*, not just the gate count: at one CX
      the measured 0.789 is already about the readout fidelity squared (0.8932²
      = 0.798), so readout, not gates, dominates at shallow depth.
    - A **calibration model**, ``0.9537**CX * 0.8932**measured``, costs nothing
      but assumes independent errors, and the ledger records the calibration
      power law coming out about twice as optimistic as the device (predicted
      49 percent against a measured 24).

    ⚠ And a measured `f` carries its own error into every number it divides:
    at 256 shots that is about ±0.04, which can exceed what the correction
    buys.

    Before any of that, the model has two premises, and both have to hold.
    They are separate questions and neither implies the other:

    1. **The error is a channel** -- memoryless, stochastic, redrawn each shot.
       A quasi-static offset is none of those, and it is what an echo
       refocuses, so an echo answers this:
       `blueqat.spin.uniform_correction_applies`. ⚠ On hardware that is a
       characterization run of its own -- a billed job, needing the same
       approval as any other.
    2. **What remains is uniform** -- not merely uncorrelated. `noise_shape`
       answers this from the measurement in hand, at no extra cost.

    A distribution can be a product of its own marginals and still be far from
    uniform: independent bits at p = 0.3 sit 0.284 away in total variation.
    Measured hardware output was 0.0395 from the product of its marginals,
    which is the correlations having gone rather than a uniform background
    having arrived. Passing the first check and failing the second still leaves
    the correction describing something that is not happening.
    """
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"rate must be in (0, 1] -- it is the surviving signal "
                         f"fraction f, not the noise -- got {rate}.")
    if rate == 1.0 or not probs or n_outcomes <= 0:
        return dict(probs)
    floor = (1.0 - rate) / n_outcomes
    return {bits: max(0.0, p - floor) / rate for bits, p in probs.items()}


def total_variation(a: Dict[Tuple[int, ...], float],
                    b: Dict[Tuple[int, ...], float]) -> float:
    """Total variation distance between two distributions over bit tuples."""
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def product_of_marginals(probs: Dict[Tuple[int, ...], float]
                         ) -> Dict[Tuple[int, ...], float]:
    """The distribution with the same single-qubit marginals and no correlation."""
    if not probs:
        return {}
    width = len(next(iter(probs)))
    ones = [sum(p for bits, p in probs.items() if bits[j]) for j in range(width)]
    out = {}
    for index in range(1 << width):
        bits = tuple((index >> j) & 1 for j in range(width))
        weight = 1.0
        for j, bit in enumerate(bits):
            weight *= ones[j] if bit else 1.0 - ones[j]
        out[bits] = weight
    return out


def noise_shape(probs: Dict[Tuple[int, ...], float]) -> Dict[str, float]:
    """How far the measured distribution is from uniform, and from a product.

    Whether `remove_uniform` is the right thing to do at all depends on what
    the noise did, and that is answerable from the measurement itself, with no
    extra job.

    - ``to_uniform`` small means the signal is mostly gone.
    - ⚠ ``to_product`` small is the case to watch. It means what survived is
      a distribution with the right single-qubit marginals and *no correlation
      between qubits* -- the noise destroyed the correlations rather than
      diluting the distribution with a uniform background. Dividing out a
      uniform component will not bring the correlations back, so a correction
      that makes the numbers look better there is making them look better
      without making them righter.

    This is not hypothetical. Analysed on real Toshiko output, the residual was
    a product distribution to within 0.0395, with single-qubit marginals of
    0.37/0.47/0.47 -- pulled toward 0.5 but not at it, which a uniform
    background cannot produce.
    """
    if not probs:
        return {"to_uniform": 0.0, "to_product": 0.0}
    width = len(next(iter(probs)))
    size = 1 << width
    uniform = {bits: 1.0 / size for bits in
               (tuple((i >> j) & 1 for j in range(width)) for i in range(size))}
    return {"to_uniform": total_variation(probs, uniform),
            "to_product": total_variation(probs, product_of_marginals(probs))}


def _warn_if_uniform_does_not_describe_it(probs, n_outcomes: int, where: str) -> None:
    """Say so when a correction is being applied to something it cannot fit.

    The model is ``q = f q_ideal + (1-f)/N``. It scales whatever correlations
    the ideal answer had by `f`; it cannot remove them. So a measured
    distribution that is close to the product of its own marginals while being
    far from uniform is something the model can only produce if the ideal
    answer was itself a product -- and if it was not, dividing out a uniform
    background is arithmetic on a premise that does not hold.

    This is checkable at exactly the moment it matters, from the counts in
    hand, for free. It warns rather than refuses because "the ideal answer was
    a product" is a thing the caller may well know and this cannot.
    """
    import warnings
    shape = noise_shape(probs)
    if shape["to_uniform"] < 1e-9:
        return                        # genuinely uniform: nothing survived
    if shape["to_product"] > 0.1 * shape["to_uniform"]:
        return                        # correlations still there; the model fits
    warnings.warn(
        f"correcting {where}, but what was measured is {shape['to_product']:.4f} "
        f"from the product of its own marginals and {shape['to_uniform']:.4f} "
        f"from uniform: the correlations are gone while the distribution is "
        f"not uniform. A uniform background cannot produce that unless the "
        f"ideal answer was itself uncorrelated -- if it was not, dividing one "
        f"out will not bring the correlations back, and the corrected numbers "
        f"will look better without being closer. Check "
        f"blueqat.spin.uniform_correction_applies for the other premise.",
        UserWarning, stacklevel=3)


class HardwareEvaluation:
    """Evaluate an ansatz's energy for one set of parameters, on a device.

    ``ansatz`` is the object the optimization used; ``circuit`` is the
    optimized circuit, normally ``VqeResult.circuit``. Nothing is submitted
    until `submit` is called with ``confirm=True``.

    Per ``~/pm/QUANTUM_HW.md`` a hardware run wants sign-off before it is
    submitted, because it spends money and monthly quota on a shared account.
    `plan()` exists so that conversation can start from a number.

    `signal_fraction` is the surviving-signal fraction `f` for the uniform-noise
    correction. Leave it None (the default) to report what was measured; check
    `noise_shape` before deciding it applies at all, and see `remove_uniform`
    for why there is no estimator for `f`.

    `before_submit(evaluation, summary)` is called after planning and before
    anything is sent; raising from it stops the submission. A checker of
    formulations, a cost ceiling, or an approval prompt goes there.
    """

    def __init__(self, ansatz, circuit: Circuit, shots: int = FREE_TIER_SHOTS,
                 qpu_id: Optional[str] = None, pack_qubits: bool = True,
                 signal_fraction: Optional[float] = None,
                 before_submit=None) -> None:
        if shots <= 0:
            raise ValueError(f"shots must be positive, got {shots}.")
        if signal_fraction is not None and not 0.0 < signal_fraction <= 1.0:
            raise ValueError("signal_fraction is the surviving fraction f in "
                             f"(0, 1], got {signal_fraction}.")
        self.ansatz = ansatz
        self.circuit = circuit
        self.shots = int(shots)
        self.qpu_id = qpu_id
        self.pack_qubits = pack_qubits
        self.signal_fraction = signal_fraction
        #: Called with (self, plan summary) after planning and before anything
        #: is sent. Raise from it to stop the submission. It is a hook rather
        #: than a built-in check so that whatever does the checking stays a
        #: dependency of the caller, not of blueqat -- which declares four
        #: dependencies and was, until today, quietly broken for anyone who
        #: did not also happen to have SciPy.
        self.before_submit = before_submit
        self._plan: Optional[List[Dict[str, Any]]] = None
        self._requests: List[Tuple[str, Tuple[int, ...], int]] = []
        self.task_ids: Dict[str, str] = {}
        self.counts: Dict[str, Dict[str, int]] = {}
        self.bit_orders: Dict[str, str] = {}

    # -- phase one: what would this cost -----------------------------------

    def plan(self) -> Dict[str, Any]:
        """Enumerate the jobs this evaluation needs. Submits nothing.

        Needs no API key, so the cost can be looked at before asking for one.
        """
        if self._plan is not None:
            return self._summary()
        recorder = _Recorder()
        self.ansatz.get_energy(self.circuit, recorder)

        jobs: Dict[str, Dict[str, Any]] = {}
        self._requests = []
        for asked, meas in recorder.requests:
            measured = _with_measurement(asked, meas)
            packed, back = (compact(measured) if self.pack_qubits
                            else (measured, {q: q for q in range(measured.n_qubits)}))
            _reject_unrunnable(packed)
            key = circuit_key(packed)
            if key not in jobs:
                jobs[key] = {"key": key, "circuit": packed,
                             "n_qubits": packed.n_qubits, "back": back, "terms": 0}
            jobs[key]["terms"] += 1
            forward = {old: new for new, old in back.items()}
            self._requests.append((key, tuple(forward[q] for q in meas),
                                   packed.n_qubits))
        self._plan = list(jobs.values())
        return self._summary()

    def _summary(self) -> Dict[str, Any]:
        assert self._plan is not None
        n_jobs = len(self._plan)
        return {
            "jobs": n_jobs,
            "shots_per_job": self.shots,
            "total_shots": n_jobs * self.shots,
            "terms": len(self._requests),
            "max_qubits": max((j["n_qubits"] for j in self._plan), default=0),
            "estimated_cost_jpy": n_jobs * (JOB_FEE_JPY + SHOT_FEE_JPY * self.shots),
            "within_free_tier": self.shots <= FREE_TIER_SHOTS,
            # Every planned job is meant to run, so every one uses a slot. A
            # job the device refuses at verification currently does not, but
            # that rule keys on whether any counts came back and is the
            # service's to change, so nothing here leans on it.
            "monthly_quota_used": n_jobs,
        }

    def quote(self, payer: str) -> dict:
        """The service's own price, rather than this module's arithmetic.

        `payer` is the wallet address that will sign, normalized to EIP-55 on
        the far side. It is not an email address or a user id.
        """
        from . import cloud
        return cloud.hardware_quote(self.shots, payer=str(payer))

    # -- phase two: submit and collect --------------------------------------

    def submit(self, confirm: bool = False) -> Dict[str, str]:
        """Send every planned job. Returns ``{circuit key: task id}``.

        Requires ``confirm=True``, for the reason `cloud.submit_hardware_job`
        does: this spends money and monthly quota on a shared account.
        """
        from . import cloud
        if self._plan is None:
            self.plan()
        assert self._plan is not None
        if not confirm:
            summary = self._summary()
            raise ValueError(
                f"submit() runs on real hardware and costs money: "
                f"{summary['jobs']} job(s), {summary['total_shots']} shots, about "
                f"{summary['estimated_cost_jpy']:.1f} JPY by the published rate, "
                f"and {summary['monthly_quota_used']} of the monthly allowance. "
                f"Get sign-off, then pass confirm=True.")
        if self.before_submit is not None:
            self.before_submit(self, self._summary())
        for job in self._plan:
            if job["key"] in self.task_ids:
                continue
            answer = cloud.submit_hardware_job(job["circuit"], shots=self.shots,
                                               qpu_id=self.qpu_id, confirm=True)
            task = answer.get("task_id") or answer.get("id")
            if not task:
                raise RuntimeError(f"the service returned no task id: {answer!r}")
            self.task_ids[job["key"]] = str(task)
        return dict(self.task_ids)

    def status(self) -> Dict[str, str]:
        """Each submitted job's state.

        A pending job has been observed answering with no status field at all,
        so a missing one is reported as ``'pending'`` rather than crashing.
        """
        from . import cloud
        out = {}
        for key, task in self.task_ids.items():
            answer = cloud.hardware_job(task, qpu_id=self.qpu_id)
            out[key] = str(answer.get("status") or 'pending')
        return out

    def ready(self) -> bool:
        """Whether every planned job has been submitted and has finished."""
        if not self.task_ids or len(self.task_ids) < len(self._plan or [1]):
            return False
        return all(s.lower() in DONE_STATES for s in self.status().values())

    def collect(self) -> Dict[str, Dict[str, int]]:
        """Fetch the counts of every finished job."""
        from . import cloud
        for key, task in self.task_ids.items():
            if key in self.counts:
                continue
            answer = cloud.hardware_job(task, qpu_id=self.qpu_id)
            state = str(answer.get("status") or '').lower()
            if state in FAILED_STATES:
                raise RuntimeError(_failure_message(task, answer))
            if state not in DONE_STATES:
                raise RuntimeError(
                    f"hardware job {task} is {state or 'pending'}, not finished. "
                    f"Jobs have been measured completing in 15 to 25 seconds, but "
                    f"the device is not always accepting work; wait() polls, and "
                    f"the task ids survive in `task_ids`.")
            result = cloud.hardware_job_result(task, qpu_id=self.qpu_id)
            counts = result.get("counts")
            if counts is None:
                raise RuntimeError(f"job {task} finished but returned no counts: {result!r}")
            self.counts[key] = unwrap_counts(counts)
            self.bit_orders[key] = parse_bit_order(result.get("bit_order"))
        return dict(self.counts)

    def wait(self, timeout: float = 900.0) -> Dict[str, Dict[str, int]]:
        """Block until every job finishes, then collect.

        Polls every 5 seconds for the first minute and then backs off: jobs have
        been measured completing in 15 to 25 seconds, so a coarse interval would
        spend minutes waiting for something already done, while a fine one would
        hammer the service through a genuinely long queue.

        Raises on timeout rather than pretending. Nothing is lost when it does:
        the task ids are in `task_ids`.
        """
        started = time.monotonic()
        deadline = started + timeout
        while True:
            states = self.status()
            if all(s.lower() in DONE_STATES for s in states.values()):
                return self.collect()
            if any(s.lower() in FAILED_STATES for s in states.values()):
                return self.collect()          # raises, with the message
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(
                    f"jobs were still {sorted(set(states.values()))} after "
                    f"{timeout:g}s. Nothing is lost: the task ids are in "
                    f"`task_ids`, and collect() picks them up later -- in this "
                    f"session, or another one through to_dict()/from_dict().")
            elapsed = now - started
            interval = 5.0 if elapsed < 60.0 else min(60.0, 5.0 + (elapsed - 60.0) / 4.0)
            time.sleep(min(interval, max(0.0, deadline - now)))

    # -- the answer ---------------------------------------------------------

    def sampler(self):
        """A sampler serving the collected counts, for `ansatz.get_energy`."""
        if not self.counts:
            raise RuntimeError("no results yet: submit(), then collect() or wait().")
        served = iter(self._requests)

        def sample(circuit: Circuit, meas) -> Dict[Tuple[int, ...], float]:
            key, packed_meas, n_qubits = next(served)
            probs = _marginal(self.counts[key], packed_meas, n_qubits,
                              self.bit_orders.get(key, "q0_first"))
            if self.signal_fraction is not None:
                _warn_if_uniform_does_not_describe_it(
                    probs, 1 << len(packed_meas), "an energy term")
                probs = remove_uniform(probs, 1 << len(packed_meas),
                                       self.signal_fraction)
            return probs

        return sample

    def energy(self) -> float:
        """The energy these measurements give.

        Evaluated through the ansatz's own `get_energy`, so the basis rotations
        and signs are the ones the optimization used rather than a second copy
        of that logic -- and the Hamiltonian's constant term is added there,
        untouched by any noise correction, which is what it should be.
        """
        if not self.counts:
            self.collect()
        return float(self.ansatz.get_energy(self.circuit, self.sampler()))

    def probabilities(self) -> Dict[Tuple[int, ...], float]:
        """The measured distribution over the circuit's own qubits.

        This is what a QAOA run wants: the cost function is diagonal, so the
        answer is read straight off the bitstrings. Keys are in the caller's own
        qubit numbering, whatever the circuit was packed into.
        """
        if not self.counts:
            self.collect()
        if self._plan is None:
            self.plan()
        assert self._plan is not None
        job = self._plan[0]
        packed = sorted(job["back"])
        probs = _marginal(self.counts[job["key"]], packed, job["n_qubits"],
                          self.bit_orders.get(job["key"], "q0_first"))
        if self.signal_fraction is not None:
            _warn_if_uniform_does_not_describe_it(
                probs, 1 << len(packed), "the measured distribution")
            probs = remove_uniform(probs, 1 << len(packed), self.signal_fraction)
        return probs

    def noise_shape(self) -> Dict[str, float]:
        """`noise_shape` of what was measured. Free, and worth reading before
        deciding whether `signal_fraction` means anything for this circuit."""
        return noise_shape(self.probabilities())

    # -- carrying the run across sessions -----------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Enough to collect the results from another session."""
        return {"task_ids": dict(self.task_ids), "counts": dict(self.counts),
                "bit_orders": dict(self.bit_orders), "shots": self.shots,
                "qpu_id": self.qpu_id, "signal_fraction": self.signal_fraction,
                "pack_qubits": self.pack_qubits}

    @classmethod
    def from_dict(cls, ansatz, circuit: Circuit,
                  state: Dict[str, Any]) -> 'HardwareEvaluation':
        """Rebuild a submitted run.

        The ansatz and circuit must be the ones that were submitted: the plan is
        recomputed from them and checked against the stored task ids, so a
        mismatch is refused rather than quietly attaching measurements to the
        wrong terms.
        """
        evaluation = cls(ansatz, circuit, shots=int(state.get("shots", FREE_TIER_SHOTS)),
                         qpu_id=state.get("qpu_id"),
                         pack_qubits=bool(state.get("pack_qubits", True)),
                         signal_fraction=state.get("signal_fraction"))
        evaluation.plan()
        assert evaluation._plan is not None
        planned = {job["key"] for job in evaluation._plan}
        if set(state.get("task_ids", {})) - planned:
            raise ValueError(
                "the stored task ids do not match this ansatz and circuit; "
                "collecting them here would attach measurements to the wrong "
                "terms.")
        evaluation.task_ids = dict(state.get("task_ids", {}))
        evaluation.counts = {k: {str(a): int(b) for a, b in v.items()}
                             for k, v in state.get("counts", {}).items()}
        evaluation.bit_orders = dict(state.get("bit_orders", {}))
        return evaluation


def _failure_message(task: str, answer: Dict[str, Any]) -> str:
    """Say why a job failed, using the message rather than the code.

    ``error_code: 101`` was measured covering at least three unrelated causes --
    mid-circuit measurement, asking for more qubits than exist, and
    ``preserve_layout`` on a device that does not support it -- so branching on
    the code tells the caller nothing. The text does.
    """
    detail = (answer.get('error_message') or answer.get('error')
              or answer.get('detail') or '')
    code = answer.get('error_code')
    return (f"hardware job {task} failed"
            + (f" (error_code {code}, which is not specific -- read the message)"
               if code else "")
            + f": {detail or 'the service gave no message'}. "
              "A job that produced no measurements is not charged for and does "
              "not currently use a monthly slot, but it does stay in the "
              "account's history as a failure.")


def _with_measurement(circuit: Circuit, meas: Sequence[int]) -> Circuit:
    """`circuit` with *every* qubit measured at the end.

    Measuring only the qubits a term needs would be enough physically, but it
    is what makes the caching useless: two terms of the same Hamiltonian ask
    for different subsets, so the circuits differ by their measurements alone
    and each becomes its own job. Measuring everything makes the circuits of
    all terms sharing a basis identical, and the extra qubits are summed out
    afterwards. For a diagonal Hamiltonian -- every QAOA cost function, where
    the rotations are empty too -- that turns one job per term into one job.

    Measuring more is free and harmless here: these measurements are terminal,
    so they disturb nothing the term cares about. `meas` is accepted to make
    the intent explicit at the call site and to assert the qubits exist.
    """
    from .circuit_funcs.flatten import flatten
    flat = flatten(circuit)
    for q in meas:
        if not 0 <= int(q) < circuit.n_qubits:
            raise ValueError(f"qubit {q} is outside a {circuit.n_qubits}-qubit circuit.")
    already = set()
    for op in flat.ops:
        if isinstance(op, Measurement):
            already.update(int(t) for t in op.target_iter(circuit.n_qubits))
    missing = [q for q in range(circuit.n_qubits) if q not in already]
    if not missing:
        return flat
    out = Circuit(circuit.n_qubits, list(flat.ops))
    for q in missing:
        out.m[q]
    return out


def _reject_unrunnable(circuit: Circuit) -> None:
    """Refuse, before submitting, what the device is known to refuse.

    The round trip teaches nothing that is not already known here, and it is
    not free of consequence even though a refused job is not charged for: it
    leaves a failure in the account's audit history that the caller could have
    been spared, and a submission carrying payment settles before the device's
    verification comes back. Whether such a job also costs a monthly slot is
    the service's own rule -- it currently does not -- so this does not depend
    on that either way.
    """
    from .backends.torch_backend import has_nonterminal_measurement
    from .circuit_funcs.flatten import flatten
    if circuit.n_qubits > MAX_HARDWARE_QUBITS:
        raise ValueError(
            f"this circuit allocates {circuit.n_qubits} qubits; the device has "
            f"{MAX_HARDWARE_QUBITS}. `n_qubits` on submission is how many to "
            f"allocate, not the highest index used, so packing the numbering "
            f"(the default) is usually enough to fit.")
    if has_nonterminal_measurement(flatten(circuit).ops, circuit.n_qubits):
        raise ValueError(
            "this circuit measures a qubit and then uses it again. OQC refuses "
            "mid-circuit measurement at submission -- verified on Toshiko, "
            "'Verification failed: No mid-circuit measurements allowed.' -- so "
            "it cannot run there. Give each measurement its own qubit and "
            "measure at the end; the device has 32, and that rewrite has been "
            "measured costing no extra CX.")
