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
"""Regressions for the findings of the cross review in issue #197.

Each test names the finding it pins. They are collected here rather than spread
across the suite because what they have in common is the failure mode: with one
exception every one of these returned a plausible wrong answer with no error.
"""

import math
import os
import stat
from collections import Counter

import pytest
import torch

from blueqat import Circuit
from blueqat.circuit_funcs.circuit_to_unitary import circuit_to_unitary
from blueqat.circuit_funcs.qasm_parser import from_qasm
from blueqat.noise import depolarizing


def _unitary(circuit):
    return torch.as_tensor(circuit_to_unitary(circuit), dtype=torch.complex128)


# --------------------------------------------------- F1: noise routing

def test_f1_shots_honours_noise():
    noise = depolarizing(0.5)
    circuit = Circuit(1).x[0].m[0]
    assert circuit.shots(1000, noise=noise, seed=1) == circuit.run(
        noise=noise, shots=1000, seed=1)


def test_f1_probs_honours_noise():
    # Depolarizing at p=0.5 on |1> leaves 3/4 weight on |1>.
    probs = Circuit(1).x[0].probs(noise=depolarizing(0.5))
    assert abs(float(probs[1]) - 0.75) < 1e-9


def test_f1_statevector_and_oneshot_refuse_noise_rather_than_ignore_it():
    for call in (lambda: Circuit(1).x[0].statevector(noise=depolarizing(0.5)),
                 lambda: Circuit(1).x[0].m[0].oneshot(noise=depolarizing(0.5))):
        with pytest.raises(ValueError):
            call()


def test_f1_noise_scale_alone_is_still_rejected_everywhere():
    with pytest.raises(ValueError):
        Circuit(1).x[0].m[0].shots(10, noise_scale=0.5)


# ------------------------------------- F2/F3: mid-circuit measurement

@pytest.mark.parametrize('backend_name', [None, 'density'])
def test_f2_f3_a_measured_value_is_not_rewritten_by_later_gates(backend_name):
    kwargs = {'shots': 1000, 'seed': 1}
    if backend_name:
        kwargs['backend'] = backend_name
    assert Circuit(1).x[0].m[0].x[0].run(**kwargs) == Counter({'1': 1000})
    assert Circuit(1).x[0].m[0].h[0].run(**kwargs) == Counter({'1': 1000})


def test_f2_a_key_no_longer_changes_the_distribution():
    plain = Circuit(1).h[0].m[0].h[0].run(shots=2000, seed=1)
    keyed = Circuit(1).h[0].m(key='a')[0].h[0].run(shots=2000, seed=1)
    assert plain == keyed
    assert 800 < plain['0'] < 1200


def test_f2_a_measured_qubit_used_as_a_control_collapses_first():
    counts = Circuit(2).h[0].m[0].cx[0, 1].m[1].run(shots=2000, seed=1)
    # Classical correlation only: the control was already collapsed.
    assert set(counts) == {'00', '11'}


def test_f2_terminal_measurement_keeps_the_fast_path():
    counts = Circuit(2).h[0].cx[0, 1].m[:].run(shots=1000, seed=1)
    assert set(counts) == {'00', '11'}


def test_f3_the_density_matrix_itself_is_still_the_average():
    # Averaging over an unread measurement is dephasing; that has not changed.
    rho = Circuit(1).h[0].m[0].run(backend='density')
    assert abs(float(rho[0, 0].real) - 0.5) < 1e-12
    assert abs(complex(rho[0, 1])) < 1e-12


# ------------------------------------------------- F4/F5/F9: QASM

def test_f4_declared_width_survives():
    assert from_qasm('qreg q[3]; creg c[3]; h q[0];').n_qubits == 3
    assert from_qasm(Circuit(3).h[0].cx[0, 1].to_qasm()).n_qubits == 3


def test_f5_a_whole_register_gate_is_applied():
    circuit = from_qasm('qreg q[2]; h q;')
    assert [op.lowername for op in circuit.ops] == ['h', 'h']
    assert circuit.run().shape[0] == 4


def test_f9_an_unbounded_exponent_is_refused():
    with pytest.raises(ValueError, match='Exponent'):
        from_qasm('qreg q[1]; rx(9**9**9) q[0];')


# ------------------------------------------------- F6: calc_u_params

@pytest.mark.parametrize('name', ['x', 'y'])
def test_f6_antidiagonal_unitaries_round_trip(name):
    from blueqat.gate import UGate
    from blueqat.utils import calc_u_params
    original = getattr(Circuit(1), name)[0].run(mode='statevector', returns='statevector')
    matrix = _unitary(getattr(Circuit(1), name)[0])
    rebuilt = UGate(0, *calc_u_params(matrix)).matrix()
    assert torch.allclose(rebuilt, matrix, atol=1e-10)
    assert original is not None


def test_f6_random_antidiagonal_unitaries_round_trip():
    from blueqat.gate import UGate
    from blueqat.utils import calc_u_params
    import random
    rng = random.Random(0)
    for _ in range(200):
        a, b = rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)
        matrix = torch.tensor([[0, complex(math.cos(a), math.sin(a))],
                               [complex(math.cos(b), math.sin(b)), 0]],
                              dtype=torch.complex128)
        assert torch.allclose(UGate(0, *calc_u_params(matrix)).matrix(), matrix, atol=1e-10)


def test_f6_decomposers_keep_the_circuit_equivalent():
    from blueqat.backends.onequbitgate_decomposer import ryrz_decomposer, u_decomposer
    from blueqat.backends.twoqubitgate_transpiler import two_qubit_gate_decompose
    for decomposer in (u_decomposer, ryrz_decomposer):
        for circuit in (Circuit(2).x[0].cz[0, 1],
                        Circuit(2).h[0].z[0].h[0].cz[0, 1]):
            out = two_qubit_gate_decompose(circuit, 'cz', mat1_decomposer=decomposer)
            before, after = _unitary(circuit), _unitary(out)
            index = int(torch.argmax(before.abs()))
            phase = after.reshape(-1)[index] / before.reshape(-1)[index]
            assert torch.allclose(before * phase, after, atol=1e-9)


# ----------------------------------------- F7/F8/F10/F15: crashes

@pytest.mark.parametrize('gate', ['swap', 'cy', 'ch'])
def test_f7_undecomposable_gates_say_so(gate):
    with pytest.raises(ValueError, match='Cannot decompose'):
        getattr(Circuit(2), gate)[0, 1].run(backend='2q_decomposition', basis='cx')


@pytest.mark.parametrize('build', [
    lambda: Circuit(3).h[:],
    lambda: Circuit(3).h[0].m[:],
    lambda: Circuit(3).h[0:2],
    lambda: Circuit(3).h[0].barrier[:],
])
def test_f8_composer_accepts_sliced_targets(build):
    build().run(backend='composer')


def test_f10_the_api_key_file_is_owner_only(tmp_path, monkeypatch):
    import blueqat.cloud as cloud
    monkeypatch.setenv("BLUEQAT_CONFIG_DIR", str(tmp_path))
    cloud.reset_configuration()
    path = cloud.save_api_key("secret")
    if os.name != 'nt':
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    cloud.reset_configuration()


def test_f15_the_documented_backend_name_works_without_arguments():
    assert isinstance(Circuit(2).cx[0, 1].run(backend='2q_decomposition'), Circuit)


# ------------------------------------- F11/F12/F13/F14/F16: behaviour

@pytest.mark.parametrize('backend_name', [None, 'density', 'stabilizer'])
def test_f11_returns_shots_always_gives_counts(backend_name):
    kwargs = {'backend': backend_name} if backend_name else {}
    result = Circuit(2).h[0].cx[0, 1].m[:].run(returns='shots', **kwargs)
    assert isinstance(result, Counter)
    assert sum(result.values()) == 1024


def test_f11_a_keyed_circuit_agrees():
    assert isinstance(Circuit(2).h[0].m(key='k')[0].run(returns='shots'), Counter)


def test_f12_asking_for_a_huge_statevector_still_raises():
    with pytest.raises(MemoryError):
        Circuit(30).h[0].m[:].run(shots=4, returns='statevector', seed=1)
    # Sampling alone is fine.
    assert sum(Circuit(30).h[0].m[:].run(shots=4, seed=1).values()) == 4


def test_f13_a_negative_ancilla_position_is_refused():
    circuit = Circuit(2).x[0].x[1]
    with pytest.raises(ValueError, match='non-negative'):
        with circuit.ancilla(pos=-1):
            pass


def test_f14_a_mixer_widens_the_ansatz():
    from blueqat.utils import QaoaAnsatz, Vqe, X, Z
    ansatz = QaoaAnsatz(1.0 * Z[0], step=1, init_circuit=Circuit(1).h[0],
                        mixer=X[1].to_expr())
    assert ansatz.n_qubits == 2
    Vqe(ansatz).run(max_iter=3)          # used to raise a dimension mismatch


@pytest.mark.parametrize('gate', ['zz', 'zzdg', 'iswap', 'iswapdg'])
def test_f16_clifford_gates_are_accepted_and_correct(gate):
    from blueqat.clifford import Clifford
    circuit = getattr(Circuit(2), gate)[0, 1]
    direct = _unitary(circuit)
    rebuilt = _unitary(Clifford.from_circuit(circuit).to_circuit())
    index = int(torch.argmax(direct.abs()))
    phase = rebuilt.reshape(-1)[index] / direct.reshape(-1)[index]
    assert torch.allclose(direct * phase, rebuilt, atol=1e-10)
    assert sum(circuit.m[:].run(backend='stabilizer', shots=8, seed=1).values()) == 8


# ------------------------------------------------------------- minor

def test_a_complex_hamiltonian_coefficient_is_refused():
    from blueqat.cloud import hamiltonian_to_terms
    from blueqat.utils import Z
    with pytest.raises(ValueError, match='Hermitian'):
        hamiltonian_to_terms((1 + 2j) * Z[0])


def test_overlapping_pulses_sharing_a_spin_are_refused():
    from blueqat.eo import from_schedule
    schedule = {"format": "blueqat-eo-schedule", "version": "1", "n_spins": 3,
                "amplitude": 1.0, "total_duration": 1.5,
                "pulses": [{"start": 0.0, "duration": 1.0, "pair": [0, 1], "theta": 1.0},
                           {"start": 0.5, "duration": 1.0, "pair": [1, 2], "theta": 1.0}]}
    with pytest.raises(ValueError, match='overlap'):
        from_schedule(schedule)
