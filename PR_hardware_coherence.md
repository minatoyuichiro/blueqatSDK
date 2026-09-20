Follows #198. Twenty-two commits, six subjects, all of them things that were found by using the SDK rather than by reading it. Suite: **2261 passed, 2 skipped, 0 failed**.

What they have in common is worth saying first: each one produced a plausible answer, or a plausible-looking failure, with nothing to indicate anything was wrong. Several were found only because somebody compared a result against theory, or deliberately broke a check to see whether it fired.

### Running a variational result on hardware

`blueqat/hardware.py`. Optimize against a simulator, then evaluate the answer once on a device. `AnsatzBase.get_energy` needed nothing new — it already rotates each Pauli term into the Z basis and calls its sampler — so the work is in what a sampler backed by a device has to survive, nearly all of it measured on OQC's Toshiko by the `tnapi` and `blueqatmcp` sessions rather than decided here.

Two phases, because hardware is not a function call. `plan()` enumerates the jobs and their cost with no API key and nothing submitted; `submit(confirm=True)` sends them; `collect()` or `wait()` brings them back. Jobs have been measured completing in 15 to 25 seconds, but the device opens twice a day on weekdays and a submission left over a weekend took 32 to 56 hours, so `to_dict`/`from_dict` carry a run into another session. The plan is obtained by running the evaluation against a recording sampler, so it cannot drift from what the real evaluation does.

Every qubit is measured, not only the ones a term needs. That sounds wasteful and is the opposite: two terms wanting different subsets produce circuits differing only in their measurements, so each becomes its own job. With all qubits measured, every term sharing a basis shares one — and a QAOA cost function is all Z, so a five-term Hamiltonian is one job at 35.6 JPY instead of five. The service does no duplicate detection of its own.

Nothing the device returns matches the simulator: counts nested under the classical register's name, a `bit_order` that is q0 *first*, keys at the submitted width, a status field that can be absent while pending, and `error_code: 101` covering at least three unrelated causes. The bit order is read from the result and refused rather than guessed if the wording changes; failures quote the message and say the code is not specific. Mid-circuit measurement and over-wide registers are refused before submission.

### Whether a noise correction may be believed

The uniform-noise correction is here, and deliberately has no estimator for its rate. Three sessions agreed on the algebra — a non-identity Pauli averages to zero over the uniform distribution, so `<P>_measured == f <P>_ideal` — and disagreed usefully about its premises, of which there are two and neither implies the other.

*Is the error a channel?* Every way of estimating `f` assumes it is: memoryless, stochastic, redrawn each shot. A quasi-static offset is none of those, and an echo is exactly the experiment that detects one. `blueqat.spin.uniform_correction_applies` runs that check; it reads 1.000 for a quasi-static offset and 0.000 for a dephasing channel. On hardware it is a billed job needing the same approval as any other.

*Is what remains uniform?* A distribution can be a product of its own marginals and still sit far from uniform — independent bits at p = 0.3 are 0.284 away — and measured Toshiko output sat 0.0395 from its own product while being 0.13 from uniform. That is the correlations having gone, not a uniform background having arrived, and dividing one out will not bring them back. `noise_shape` reports both distances from the measurement in hand, for free, and the correction warns when it is being applied to something it cannot fit.

A density-matrix study found `f_mirror == f**2` exact under global depolarizing noise, which would make a mirror circuit a good way to measure `f`. The device does not support extrapolating that way: Bell-plus-CX fidelities of 0.789, 0.711, 0.742 and 0.539 at 1, 3, 5 and 9 CX are not monotonic. Where a noise model and the hardware disagree, this takes the hardware.

### Coherence experiments

`blueqat/spin.py`. `QuasiStatic` existed; the experiments that make it useful did not. A single T2* is a number both a quasi-static offset and a dephasing channel explain — only the shape of the curve and the response to refocusing tell them apart. Ramsey, Hahn echo and CPMG, with `identify_noise` reading the shape off the curve and `refocusing_gain` doing it in one call.

Both sides were checked against theory. A quasi-static offset decays a Ramsey fringe as `exp(-(sigma t)**2 / 2)` to within Monte Carlo error, and an echo returns the coherence to 1.000000000 at every delay, since the offset is fixed within a shot and the phase either side of the pulse cancels exactly. A dephasing channel decays exponentially and its echo comes out slightly *worse*, the refocusing pulse being one more gate. Two other sessions reproduced both independently, agreeing to 0.0006 and matching even the amount by which the echo is worse.

Two details would have been wrong silently. Idle time is spent on `rz(0)` gates rather than barriers: both end a layer, so a quasi-static phase accumulates identically, but a `NoiseModel` attaches channels to gates — idling on barriers leaves the waiting noiseless and a Markovian T2 curve comes out flat, the qubit appearing immortal while it waits. The first version did that. And a sequence with `delay` idle layers evolves for `delay + 1` of them, confirmed against the decay at 40000 samples rather than taken from a docstring.

### Which end is qubit 0

The costly one. A session working thirty benchmark problems read a distribution backwards and answered one wrong; the other twenty-nine passed because their states were symmetric, which is to say the mistake was invisible everywhere it did not matter.

`run(shots=...)` took a `bit_order` argument and documented the convention. `probs()` documented it but had no argument. `statevector()`, at eighty-three characters, had neither — and was worse than that: it *accepted* `bit_order`, passed it down, had it validated by the backend, and ignored it, so asking for the other convention returned the default one with nothing said. All three now take the same argument with the same values, and all three say which end is qubit 0.

`blueqat.BIT_ORDER` states the convention, because it has not always been this one: blueqat 2.0.4's numpy and numba backends put qubit 0 at the left. It is not sufficient alone — it was added after 2.1.3 shipped, so a correct 2.1.3 lacks it, and reading absence as "old and wrong" is the same mistake inverted — so `measure_bit_order()` establishes it by running a circuit, and its docstring carries the two lines to copy for code that predates even that.

### Text, and what a model reads

QASM arrives from MCP clients, which means it arrives pasted, and what gets pasted out of a PDF is full-width punctuation or one of the 214 Kangxi radicals, which render identically to ordinary characters. The parser already refused such a program rather than misreading it; what was missing is that the error named nothing a reader could see. Failures now name the characters with their code points, and `normalize=True` applies NFKC — opt in, because rewriting input silently is a guess.

The public helpers are split by what may be done with a finding, which took measuring to get right. A Kangxi radical says what it is in its own Unicode name, so one in prose was chosen by nobody. A compatibility ideograph says only `CJK COMPATIBILITY IDEOGRAPH-FA10`, and normalizes onto characters that appear in people's names — a document carrying one may be spelling a name correctly. `unfixable_lookalikes` reports the twelve that NFKC leaves alone, so "normalized" is never reported as "resolved".

Separately, MCP tools raise `NotExamined` when a service fails, because the caller is a model and a bare failure reads as a verdict on the input: shown a connection error it rewrites a circuit that was never looked at. `CloudOutcomeUnknown` passes through unwrapped — the wrapper advises retrying, which is the one thing not to do when the work may already have run and been paid for.

### Smaller

An exact four-CX two-qubit basis, down from six: one CX sandwich carries an XX and a ZZ term at once, and the three terms commute, so they split into two sandwiches. Over a Shannon decomposition that is 36 CX to 28 at three qubits and 168 to 136 at four. The arrangement was found by enumerating candidates in blueqat's own gate conventions rather than recalled — two earlier guesses were wrong.

A Cloudflare 403 now says it is a Cloudflare 403. Measured one variable at a time: no User-Agent header is 403 because urllib supplies `Python-urllib` itself, an empty one is 200, a one-character one is 200. The rule is about that string, and the request never reached the service, so the answer is not about the API key.

The default backend is `tensornet`, whose cost follows how a circuit is wired. A 19-qubit circuit was measured needing 16 GiB against 8 MiB for its dense vector; unbounded that is a SIGKILL, which leaves no traceback. Contractions are now priced before running, and the session that hit it confirmed the predicted 16.0 GiB against an actual allocation failure at 17179869184 bytes.

And four things a first-time user got stuck on: `expect` on a dict raising `AttributeError` about `simplify` from three frames down; `Circuit().expect(Z[0])` raising about a zero-qubit state when it is the first expectation anyone measures; the Pauli operators living only in a module named for miscellany, which was measured costing fifteen minutes to find; and the bit-order gap above.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01WtTQjVhLF6XmcVDyMPrMAK
