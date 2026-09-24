# Changelog

## 2.2.0

2.1.3 was cut on 2026-07-15 and then carried 17,666 added lines across 113
files, including nine new modules, without the version moving. Anything
installed in that window called itself 2.1.3, so "which 2.1.3?" became a
question that could only be answered by reading `direct_url.json` by hand —
and it was asked, more than once, to settle a bug report one session could
reproduce and another could not.

So two things change together: the number is correct again, and the number is
no longer the only thing to go on.

### Saying which build this is

`blueqat.version_info()` reports the version, the commit, where that commit
came from (`'checkout'`, `'install'` or `None`), and the path being imported.
`blueqat.installed_revision()` is the commit alone.

The commit is decided from the file that is actually imported, not from
whichever distribution metadata is found first. A source checkout carries a
`blueqat.egg-info` that shadows an installed `dist-info` whenever the working
directory is the repository; before that was handled, the same call answered
differently depending on where it was run from, and could have reported an
installed commit while running checkout code. A checkout appends `-dirty` when
the working tree differs, because the commit is not the whole truth while
somebody is editing. A release from PyPI records no commit and answers `None`
— an answer, not a failure.

### Two behaviour changes

Both replace something that was silently wrong, and both can make code that
previously ran raise. That is the intent: the previous behaviour produced a
result for a question nobody asked.

- **A gate written without its qubits no longer disappears.**
  `circuit.ry(theta, 0)` is how several other toolkits are written; blueqat
  spells the qubit as a subscript. The call used to build a gate and throw it
  away, silently. It now warns, naming the call that was meant
  (`ry(0.3)[0]`). Subscripting it as well — `ry(theta, q)[k]`, or
  `ry[k](theta)` — raises, because there the mistake used to survive into a
  circuit that looked right. Running a circuit while a gate is still waiting
  for its qubits warns too, which is the case a notebook creates by holding
  the last expression of every cell.

- **`statevector(bit_order=...)` and `probs(bit_order=...)` are honoured.**
  `statevector` previously accepted the argument, validated it, and ignored
  it. `probs` had no such argument while `run(shots=...)` did, and the gap
  itself was the trap: "I checked with `run()`" was not an answer about the
  other two. `blueqat.BIT_ORDER` states the convention, and
  `blueqat.measure_bit_order()` establishes it by running a circuit, for code
  that must also work against versions predating the constant.

### New

- `blueqat.noise` — Kraus channels, `NoiseModel`, and `QuasiStatic`, which is
  not a channel and refocuses under an echo.
- `blueqat.spin` — Ramsey, Hahn echo and CPMG, and telling a quasi-static
  offset apart from a dephasing channel by the shape of the curve.
- `blueqat.stabilizer`, `blueqat.clifford` — tableau simulation and Clifford
  sampling.
- `blueqat.qec` — stabilizer codes, syndrome extraction, matching decoder.
- `blueqat.decompose` — arbitrary unitaries into circuits: exact two-qubit
  (four CX with `basis='cx'`, six with rotations), optimal three-CX by fit,
  Quantum Shannon for any width, isometries.
- `blueqat.hardware` — running a VQE or QAOA result on a device, in two
  phases, with the cost known before anything is submitted.
- `blueqat.cloud` — the hosted API, including telling an unknown outcome from
  a failure when a gateway times out.
- `blueqat.backends.cuquantum_backend` — cuStateVec, with a CPU mode that is
  the same translation and is what the tests exercise.
- `blueqat.eo` — exchange-only spin qubits; `shortest=True` solves one-qubit
  gates in closed form instead of composing tables.
- Top-level exports for the things people could not find: `I`, `X`, `Y`, `Z`,
  `Expr`, `Term`, `parse_hamiltonian`, `from_qubo`, `ground_state_energy`,
  `exhaustive_minimum`.
- `Circuit.expect` accepts a string, refuses a dict with an error that says
  what a Hamiltonian is, and works on a circuit narrower than the Hamiltonian
  — `Circuit().expect(Z[0])` is 1, not an error.
- An unknown keyword to `run` is reported with the near miss, rather than
  accepted and dropped.
