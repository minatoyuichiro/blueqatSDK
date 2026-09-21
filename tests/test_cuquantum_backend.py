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
"""The cuQuantum connector, on a machine with no GPU.

What is tested here is the translation -- which matrix, on which qubits, in
what order -- by running it with the CPU applier and requiring the result to
equal `TorchBackend`'s. That is the same code path the device takes, so a
disagreement here would be a disagreement there. What is *not* tested is the
cuStateVec calls themselves, which need a GPU.
"""

import random
import warnings

import pytest
import torch

import blueqat.backends.cuquantum_backend as cuquantum
from blueqat import Circuit

CPU = dict(device='cpu')


def _state(circuit):
    return torch.as_tensor(circuit.run(backend='statevector'),
                           dtype=torch.complex128)


# --- the translation, against the simulator it must agree with -------------

@pytest.mark.parametrize('build', [
    lambda: Circuit(1).h[0],
    lambda: Circuit(1).rx(0.3)[0].ry(0.7)[0].rz(1.1)[0],
    lambda: Circuit(2).x[0].cx[0, 1],
    lambda: Circuit(2).h[0].cx[0, 1],
    lambda: Circuit(2).h[1].cx[1, 0],               # the reversed control
    lambda: Circuit(3).h[0].cx[0, 1].cx[1, 2],
    lambda: Circuit(3).h[0].h[1].ccx[0, 1, 2],
    lambda: Circuit(3).h[0].rx(0.4)[1].swap[0, 2],
    lambda: Circuit(3).h[0].h[1].cswap[0, 1, 2],
    lambda: Circuit(2).h[0].h[1].cz[0, 1],
    lambda: Circuit(4).h[0].cx[0, 1].ry(0.3)[2].ccx[0, 1, 3].cz[2, 3],
])
def test_it_gives_the_same_state_as_the_cpu_simulator(build):
    circuit = build()
    got = circuit.run(backend='cuquantum', **CPU)
    assert torch.allclose(got, _state(circuit), atol=1e-12)


def test_random_circuits_agree_too():
    """The qubit order of a multi-qubit matrix is the thing most likely to be
    wrong, and wrong quietly: on a symmetric state the two orders agree."""
    generator = random.Random(0)
    for n_qubits in (2, 3, 4):
        for _ in range(6):
            circuit = Circuit(n_qubits)
            for _ in range(8):
                kind = generator.choice(['h', 'x', 'y', 'z', 's', 't', 'rx',
                                         'ry', 'rz', 'cx', 'cz', 'swap',
                                         'ccx', 'cswap'])
                if kind in ('cx', 'cz', 'swap'):
                    a, b = generator.sample(range(n_qubits), 2)
                    getattr(circuit, kind)[a, b]
                elif kind in ('ccx', 'cswap'):
                    if n_qubits < 3:
                        continue
                    a, b, c = generator.sample(range(n_qubits), 3)
                    getattr(circuit, kind)[a, b, c]
                elif kind in ('rx', 'ry', 'rz'):
                    getattr(circuit, kind)(generator.uniform(0, 6))[
                        generator.randrange(n_qubits)]
                else:
                    getattr(circuit, kind)[generator.randrange(n_qubits)]
            got = circuit.run(backend='cuquantum', **CPU)
            assert torch.allclose(got, _state(circuit), atol=1e-11)


def test_the_qubit_order_of_a_two_qubit_matrix_is_the_measured_one():
    """`CXGate.matrix()` is the 4x4, and it expects (target, control) -- the
    reverse of how the gate lists them. Taken from a comparison rather than
    from reasoning about index conventions, because the wrong choice is off
    by 0.707 and silent."""
    circuit = Circuit(2).h[0].rx(0.7)[1].cx[0, 1]
    assert torch.allclose(circuit.run(backend='cuquantum', **CPU),
                          _state(circuit), atol=1e-12)
    gate = Circuit(2).cx[0, 1].ops[0]
    assert cuquantum._qubits_for_matrix(gate, 2) == [(1, 0)]


# --- the conventions it has to share ---------------------------------------

def test_counts_match_the_cpu_backend_shot_for_shot():
    """Sampling happens on the host with the same generator, so a seeded run
    gives the same counts -- not merely the same distribution."""
    circuit = Circuit(3).h[0].cx[0, 1].m[:]
    assert (dict(circuit.run(backend='cuquantum', shots=4000, seed=7, **CPU))
            == dict(circuit.run(shots=4000, seed=7)))


@pytest.mark.parametrize('bit_order,expected', [('q0_last', '001'),
                                                ('q0_first', '100')])
def test_it_honours_bit_order(bit_order, expected):
    circuit = Circuit(3).x[0].m[:]
    counts = circuit.run(backend='cuquantum', shots=4, seed=1,
                         bit_order=bit_order, **CPU)
    assert list(counts) == [expected]


def test_an_unsupported_return_is_refused():
    with pytest.raises(ValueError, match='not supported'):
        Circuit(2).h[0].run(backend='cuquantum', returns='samples', **CPU)


def test_gate_matrices_come_from_the_gates_themselves():
    """Rewriting them in the connector is how it drifts from the simulator."""
    import inspect
    source = inspect.getsource(cuquantum)
    assert 'gate.matrix()' in source


# --- saying where it is actually running -----------------------------------

def test_falling_back_to_the_cpu_says_so():
    """A silent fallback turns a GPU benchmark into a CPU one without telling
    anybody, which is the whole failure this guards against."""
    with pytest.warns(RuntimeWarning, match='running on the CPU'):
        Circuit(2).h[0].run(backend='cuquantum')


def test_asking_for_the_cpu_is_quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        Circuit(2).h[0].run(backend='cuquantum', device='cpu')


def test_asking_for_cuda_fails_rather_than_falling_back():
    if torch.cuda.is_available():                       # pragma: no cover
        pytest.skip('this machine has a GPU; the fallback is not exercised')
    with pytest.raises(cuquantum.CuQuantumNotAvailable) as exc:
        Circuit(2).h[0].run(backend='cuquantum', device='cuda')
    message = str(exc.value)
    assert 'cuquantum-python' in message
    assert "backend='statevector'" in message           # and what to do instead


def test_an_unknown_device_is_refused():
    with pytest.raises(ValueError, match="device must be"):
        cuquantum.CuQuantumBackend(device='tpu')


def test_both_names_are_registered():
    from blueqat.backends.backendbase import get_backend
    for name in ('cuquantum', 'custatevec'):
        assert isinstance(get_backend(name), cuquantum.CuQuantumBackend)
