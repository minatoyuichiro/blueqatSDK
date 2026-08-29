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
"""`Circuit.exp_pauli` (Pauli-product time evolution) and matrix-free expectation."""

import math

import pytest
import torch

from blueqat import Circuit
from blueqat.utils import I, X, Y, Z, pauli_expectation


def _state(n, prep=None):
    c = Circuit(n).h[:].t[0]
    if prep is not None:
        prep(c)
    return c


def _matrix_reference(hamiltonian, statevector, n_qubits):
    """The pre-existing way: build H as a dense matrix and take <psi|H|psi>."""
    h = hamiltonian.to_expr().simplify().to_matrix(n_qubits).to(statevector.dtype)
    return torch.vdot(statevector, h @ statevector).real


# --------------------------------------------------------------- exp_pauli

@pytest.mark.parametrize('paulis,term,n', [
    ({0: 'Z'}, 1.0 * Z[0], 1),
    ({0: 'X'}, 1.0 * X[0], 1),
    ({0: 'Y'}, 1.0 * Y[0], 1),
    ({0: 'Z', 1: 'Z'}, 1.0 * Z[0] * Z[1], 2),
    ({0: 'X', 1: 'Y'}, 1.0 * X[0] * Y[1], 2),
    ({0: 'X', 1: 'X', 2: 'Z', 3: 'Y'}, 1.0 * X[0] * X[1] * Z[2] * Y[3], 4),
    ({2: 'Z', 0: 'X'}, 1.0 * X[0] * Z[2], 3),
])
def test_exp_pauli_matches_the_closed_form(paulis, term, n):
    # P**2 == I, so exp(-i t P) is exactly cos(t) - i sin(t) P.
    t = 0.37
    psi0 = _state(n).run()
    got = _state(n, lambda c: c.exp_pauli(paulis, t)).run()
    p_mat = term.to_expr().simplify().to_matrix(n).to(psi0.dtype)
    want = math.cos(t) * psi0 - 1j * math.sin(t) * (p_mat @ psi0)
    assert torch.allclose(got, want, atol=1e-10)


def test_exp_pauli_agrees_with_get_time_evolution():
    # The two must not drift apart: same operator, same convention.
    term = (1.0 * X[0] * Z[1] * Y[2]).to_term()
    t = 0.4

    by_macro = Circuit(3).h[:].exp_pauli({0: 'X', 1: 'Z', 2: 'Y'}, t).run()
    evolved = Circuit(3).h[:]
    term.get_time_evolution()(evolved, t)
    assert torch.allclose(by_macro, evolved.run(), atol=1e-12)


def test_exp_pauli_single_z_is_rz_of_twice_theta():
    t = 0.3
    assert torch.allclose(Circuit(1).h[0].exp_pauli({0: 'Z'}, t).run(),
                          Circuit(1).h[0].rz(2 * t)[0].run(), atol=1e-12)


def test_exp_pauli_is_unitary_and_chains():
    c = Circuit(3).h[:].exp_pauli({0: 'X', 2: 'Z'}, 0.3).exp_pauli({1: 'Y'}, -0.2)
    assert abs(float(torch.linalg.norm(c.run())) - 1.0) < 1e-10


def test_exp_pauli_inverse_undoes_it():
    psi0 = Circuit(3).h[:].t[0].run()
    back = Circuit(3).h[:].t[0].exp_pauli({0: 'X', 1: 'Z'}, 0.6).exp_pauli({0: 'X', 1: 'Z'}, -0.6)
    assert torch.allclose(back.run(), psi0, atol=1e-12)


def test_exp_pauli_letters_are_case_insensitive():
    a = Circuit(2).h[:].exp_pauli({0: 'x', 1: 'z'}, 0.25).run()
    b = Circuit(2).h[:].exp_pauli({0: 'X', 1: 'Z'}, 0.25).run()
    assert torch.allclose(a, b, atol=1e-12)


def test_exp_pauli_ignores_identity_factors():
    a = Circuit(3).h[:].exp_pauli({0: 'X', 1: 'I', 2: 'Z'}, 0.25).run()
    b = Circuit(3).h[:].exp_pauli({0: 'X', 2: 'Z'}, 0.25).run()
    assert torch.allclose(a, b, atol=1e-12)


def test_exp_pauli_of_pure_identity_appends_nothing():
    # exp(-i t I) is a global phase, which a statevector does not carry.
    assert Circuit(2).exp_pauli({0: 'I'}, 0.5).ops == []
    assert Circuit(2).exp_pauli({}, 0.5).ops == []


def test_exp_pauli_key_order_does_not_matter():
    a = Circuit(3).h[:].exp_pauli({2: 'Z', 0: 'X', 1: 'Y'}, 0.3).run()
    b = Circuit(3).h[:].exp_pauli({0: 'X', 1: 'Y', 2: 'Z'}, 0.3).run()
    assert torch.allclose(a, b, atol=1e-12)


def test_exp_pauli_widens_the_circuit():
    assert Circuit().exp_pauli({0: 'X', 3: 'Z'}, 0.1).n_qubits == 4


def test_exp_pauli_is_differentiable():
    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    # <Z0> after exp(-i theta X0) on |0> is cos(2 theta); d/dtheta = -2 sin(2 theta).
    energy = Circuit(1).exp_pauli({0: 'X'}, theta).expect(1.0 * Z[0])
    energy.backward()
    assert abs(float(energy.detach()) - math.cos(2 * 0.3)) < 1e-10
    assert abs(float(theta.grad) + 2 * math.sin(2 * 0.3)) < 1e-9


@pytest.mark.parametrize('bad', [{-1: 'X'}, {0: 'Q'}, {'a': 'X'}, {0: 'XY'}])
def test_exp_pauli_rejects_bad_specifications(bad):
    with pytest.raises(ValueError):
        Circuit(2).exp_pauli(bad, 0.1)


# ------------------------------------------------------- pauli_expectation

@pytest.mark.parametrize('hamiltonian', [
    1.0 * Z[0],
    1.0 * X[0],
    1.0 * Y[0],
    2.5 * I,
    1.0 * X[0] * X[1] * Z[2] * Y[3],
    1.0 * Z[0] * Z[1] - 0.5 * X[1],
    1.23 * Z[0] + 4.56 * X[1] * Z[2] + 0.7 * Y[3],
    (1.0 * X[0] * Y[0]).to_expr(),
])
@pytest.mark.parametrize('prep', ['h', 'entangled', 'rotated'])
def test_expectation_matches_the_matrix_form(hamiltonian, prep):
    circuits = {
        'h': Circuit(4).h[:],
        'entangled': Circuit(4).h[0].cx[0, 1].cx[1, 2].t[3],
        'rotated': Circuit(4).ry(0.7)[0].rx(1.1)[1].rz(0.3)[2].h[3],
    }
    psi = circuits[prep].run()
    assert torch.allclose(pauli_expectation(hamiltonian, psi, 4),
                          _matrix_reference(hamiltonian, psi, 4), atol=1e-10)


def test_circuit_expect_uses_the_matrix_free_path():
    h = 1.0 * Z[0] * Z[1] + 0.5 * X[2]
    c = Circuit(4).h[0].cx[0, 1].ry(0.4)[2]
    assert torch.allclose(c.expect(h), _matrix_reference(h, c.run(), 4), atol=1e-10)


def test_run_with_hamiltonian_is_unchanged():
    assert abs(float(Circuit(4).x[:].run(hamiltonian=1 * Z[0] + 1 * Z[1])) + 2.0) < 1e-12


def test_expectation_scales_past_the_matrix_limit():
    # Building the 2**n x 2**n matrix for n=18 would need ~1TB; term-by-term is
    # a handful of passes over the statevector.
    value = Circuit(18).h[:].expect(1.0 * Z[0] * Z[1] + 0.5 * X[2])
    assert abs(float(value) - 0.5) < 1e-10


def test_expectation_of_identity_is_the_coefficient():
    assert abs(float(Circuit(3).h[:].expect(2.5 * I)) - 2.5) < 1e-12


def test_expectation_is_real_for_hermitian_hamiltonians():
    value = Circuit(3).h[0].cx[0, 1].expect(1.0 * Y[0] * Y[1] + 1.0 * X[2])
    assert not torch.is_complex(value)


def test_expectation_is_differentiable_in_the_state():
    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    energy = Circuit(2).ry(theta)[0].expect(1.0 * Z[0])
    energy.backward()
    assert abs(float(energy.detach()) - math.cos(0.3)) < 1e-10
    assert abs(float(theta.grad) + math.sin(0.3)) < 1e-9


def test_expectation_is_differentiable_in_the_coefficients():
    coeff = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    energy = Circuit(1).x[0].expect(coeff * Z[0])
    energy.backward()
    assert abs(float(energy.detach()) + 2.0) < 1e-10
    assert abs(float(coeff.grad) + 1.0) < 1e-10


def test_expectation_rejects_a_hamiltonian_wider_than_the_state():
    psi = Circuit(2).h[:].run()
    with pytest.raises(ValueError):
        pauli_expectation(1.0 * Z[5], psi, 2)


def test_expectation_rejects_a_mismatched_width():
    psi = Circuit(3).h[:].run()
    with pytest.raises(ValueError):
        pauli_expectation(1.0 * Z[0], psi, 2)


def test_expectation_defaults_to_the_implied_width():
    psi = Circuit(3).h[0].cx[0, 1].run()
    assert torch.allclose(pauli_expectation(1.0 * Z[0] * Z[1], psi),
                          pauli_expectation(1.0 * Z[0] * Z[1], psi, 3), atol=1e-12)


def test_expectation_of_an_empty_hamiltonian_is_zero():
    psi = Circuit(2).h[:].run()
    assert abs(float(pauli_expectation(1.0 * Z[0] - 1.0 * Z[0], psi, 2))) < 1e-12


def test_exp_pauli_and_expectation_reproduce_a_trotter_step():
    # A one-term Trotter step then measuring the same term: <P> after exp(-i t P)
    # is unchanged (they commute), a check that the two APIs share a convention.
    h = 1.0 * Z[0] * X[1]
    before = Circuit(2).h[:].t[0].expect(h)
    after = Circuit(2).h[:].t[0].exp_pauli({0: 'Z', 1: 'X'}, 0.42).expect(h)
    assert torch.allclose(before, after, atol=1e-10)
