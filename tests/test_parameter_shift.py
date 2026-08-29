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
"""The parameter-shift rule, and the shot-based VQE it makes possible."""

import pytest
import torch

from blueqat import Circuit
from blueqat.utils import (AnsatzBase, QaoaAnsatz, Vqe, X, Y, Z,
                           get_measurement_sampler, non_sampling_sampler,
                           parameter_shift_gradient, qubo_bit as q)

_H2 = (1.0 * Z[0] * Z[1] + 0.5 * X[0] + 0.3 * Z[1]).simplify()


class _Ansatz(AnsatzBase):
    """Wraps a plain builder function so tests can state circuits inline."""

    def __init__(self, build, hamiltonian, n_params):
        super().__init__(hamiltonian, n_params)
        self._build = build

    def get_circuit(self, params):
        return self._build(params)


def _exact_gradient(ansatz, params, hamiltonian):
    tracked = params.clone().requires_grad_(True)
    ansatz.get_circuit(tracked).expect(hamiltonian).backward()
    return tracked.grad


def _shift_gradient(ansatz, params):
    return parameter_shift_gradient(
        ansatz, params, lambda c: ansatz.get_energy(c, non_sampling_sampler))


# ---------------------------------------------- the rule against exact gradients

@pytest.mark.parametrize('build,n_params,params', [
    # plain, one parameter per gate
    (lambda p: Circuit(2).ry(p[0])[0].rx(p[1])[1].cx[0, 1], 2, [0.31, 1.2]),
    # one parameter feeding two gates: the chain rule has to sum both
    (lambda p: Circuit(2).ry(p[0])[0].cx[0, 1].ry(p[0])[1].rx(p[1])[0], 2, [0.63, 1.1]),
    # a sliced target is several gate applications written as one operation
    (lambda p: Circuit(2).h[:].rx(p[0])[:].cx[0, 1].rz(p[1])[1], 2, [0.4, 0.9]),
    # two-qubit rotations
    (lambda p: Circuit(2).h[0].rzz(p[0])[0, 1].ryy(p[1])[0, 1], 2, [0.5, 0.25]),
    (lambda p: Circuit(2).h[:].rxx(p[0])[0, 1].cp(p[1])[0, 1], 2, [1.3, 0.6]),
    # phase gates
    (lambda p: Circuit(2).h[:].p(p[0])[0].phase(p[1])[1].cx[0, 1], 2, [0.7, 2.2]),
])
def test_shift_rule_matches_exact_gradient(build, n_params, params):
    ansatz = _Ansatz(build, _H2, n_params)
    p = torch.tensor(params, dtype=torch.float64)
    _, grad = _shift_gradient(ansatz, p)
    assert torch.allclose(grad, _exact_gradient(ansatz, p, _H2), atol=1e-9)


def test_shift_rule_reports_the_energy_too():
    ansatz = _Ansatz(lambda p: Circuit(2).ry(p[0])[0].cx[0, 1].rx(p[1])[1], _H2, 2)
    p = torch.tensor([0.2, 0.9], dtype=torch.float64)
    value, _ = _shift_gradient(ansatz, p)
    assert abs(float(value) - float(ansatz.get_circuit(p).expect(_H2))) < 1e-10


def test_shift_rule_reaches_gates_inside_a_named_block():
    def build(p):
        c = Circuit(2)
        with c.block('layer'):
            c.ry(p[0])[0].cx[0, 1]
        c.rz(p[1])[1]
        return c
    ansatz = _Ansatz(build, _H2, 2)
    p = torch.tensor([0.8, 0.2], dtype=torch.float64)
    _, grad = _shift_gradient(ansatz, p)
    assert torch.allclose(grad, _exact_gradient(ansatz, p, _H2), atol=1e-9)


@pytest.mark.parametrize('step', [1, 2])
def test_shift_rule_matches_exact_gradient_for_qaoa(step):
    # Every QAOA angle drives many gates at once, including sliced ones.
    hamiltonian = (q(0) - q(1) + 2 * q(0) * q(1)).simplify()
    ansatz = QaoaAnsatz(hamiltonian, step)
    p = torch.linspace(0.3, 1.1, 2 * step, dtype=torch.float64)
    _, grad = _shift_gradient(ansatz, p)
    assert torch.allclose(grad, _exact_gradient(ansatz, p, ansatz.hamiltonian), atol=1e-9)


# ------------------------------------------------------------------- refusals

@pytest.mark.parametrize('gate', ['crx', 'cry', 'crz'])
def test_controlled_rotations_are_refused_not_approximated(gate):
    # Their generators have four eigenvalues, so the two-term rule is simply wrong
    # for them -- a silent wrong gradient would be far worse than an error.
    build = lambda p: getattr(Circuit(2).h[0], gate)(p[0])[0, 1]
    with pytest.raises(ValueError, match='two-term'):
        _shift_gradient(_Ansatz(build, _H2, 1), torch.tensor([0.4], dtype=torch.float64))


def test_multi_parameter_gates_are_refused():
    build = lambda p: Circuit(1).u(p[0], 0.2, 0.3)[0]
    with pytest.raises(ValueError):
        _shift_gradient(_Ansatz(build, 1.0 * Z[0], 1), torch.tensor([0.4], dtype=torch.float64))


def test_an_ansatz_that_drops_the_tensor_is_reported():
    # float() inside get_circuit severs the link the chain rule needs.
    build = lambda p: Circuit(2).ry(float(p[0]))[0].cx[0, 1]
    with pytest.raises(ValueError, match='No gate parameter'):
        _shift_gradient(_Ansatz(build, _H2, 1), torch.tensor([0.4], dtype=torch.float64))


# ------------------------------------------------------------------------ Vqe

def _qubo():
    return (q(0) - q(1)).simplify()


def test_shot_based_vqe_runs_and_converges():
    # This is what used to fail outright with "element 0 of tensors does not
    # require grad": sampling discards the graph backpropagation needs.
    vqe = Vqe(QaoaAnsatz(_qubo(), 1), sampler=get_measurement_sampler(2000, seed=3),
              seed=42)
    result = vqe.run(max_iter=60)
    assert result.most_common(1)[0][0] == (0, 1)
    assert vqe.sampler_call_count > 0
    assert result.loss_history[-1] < result.loss_history[0]


def test_auto_uses_backprop_when_the_objective_is_differentiable():
    # No sampler: the exact path still backpropagates, so a run costs one
    # objective evaluation per iteration rather than two per gate.
    exact = Vqe(QaoaAnsatz(_qubo(), 1), seed=42).run(max_iter=20)
    forced = Vqe(QaoaAnsatz(_qubo(), 1), seed=42).run(max_iter=20, gradient='backprop')
    assert torch.allclose(exact.params, forced.params)


def test_parameter_shift_and_backprop_agree_on_the_exact_path():
    shift = Vqe(QaoaAnsatz(_qubo(), 1), seed=42).run(max_iter=15,
                                                     gradient='parameter_shift')
    backprop = Vqe(QaoaAnsatz(_qubo(), 1), seed=42).run(max_iter=15,
                                                        gradient='backprop')
    assert torch.allclose(shift.params, backprop.params, atol=1e-8)


def test_backprop_still_fails_loudly_on_a_shot_sampler():
    vqe = Vqe(QaoaAnsatz(_qubo(), 1), sampler=get_measurement_sampler(200, seed=1))
    with pytest.raises(RuntimeError):
        vqe.run(max_iter=3, gradient='backprop')


def test_shot_based_vqe_records_a_loss_history():
    result = Vqe(QaoaAnsatz(_qubo(), 1), sampler=get_measurement_sampler(500, seed=2),
                 seed=7).run(max_iter=8)
    assert len(result.loss_history) == 8
    assert all(isinstance(v, float) for v in result.loss_history)


def test_shot_based_vqe_is_reproducible_with_a_seed():
    def run():
        return Vqe(QaoaAnsatz(_qubo(), 1), sampler=get_measurement_sampler(400, seed=5),
                   seed=13).run(max_iter=6)
    assert torch.allclose(run().params, run().params)


def test_gradient_option_is_validated():
    with pytest.raises(ValueError):
        Vqe(QaoaAnsatz(_qubo(), 1), gradient='finite_difference')
    with pytest.raises(ValueError):
        Vqe(QaoaAnsatz(_qubo(), 1)).run(max_iter=1, gradient='finite_difference')
