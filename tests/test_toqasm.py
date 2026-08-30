# Copyright 2019 The Blueqat Developers
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

import math

import numpy as np
import pytest

from blueqat import Circuit
from blueqat.circuit_funcs import from_qasm

QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
h q[1];
cx q[0],q[1];
rz(1.23) q[2];
x q[2];
y q[2];
cz q[2],q[1];
z q[1];
ry(4.56) q[0];
u(1.0,2.0,3.0) q[0];
cu(2.0,3.0,1.0,0.5) q[2],q[0];
reset q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];"""

def test_qasm1():
    c = Circuit()
    c.h[0, 1].cx[0, 1].rz(1.23)[2].x[2].y[2].cz[2, 1].z[1].ry(4.56)[0]
    c.u(1.0, 2.0, 3.0)[0]
    c.cu(2.0, 3.0, 1.0, 0.5)[2, 0]
    c.reset[1]
    qasm = c.m[:].to_qasm()
    assert QASM == qasm

def qasm_prologue(n_qubits):
    return "\n".join([
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        "qreg q[" + str(n_qubits) + "];",
        "creg c[" + str(n_qubits) + "];"
    ])

def test_qasm_nocache():
    correct_qasm = qasm_prologue(1) + "\nx q[0];\ny q[0];\nz q[0];"
    c = Circuit().x[0].y[0].z[0]
    c.run()
    c.to_qasm()
    qasm = c.to_qasm()
    assert qasm == correct_qasm

def test_qasm_noprologue():
    correct_qasm = "x q[0];\ny q[0];\nz q[0];"
    c = Circuit().x[0].y[0].z[0]
    qasm = c.to_qasm(output_prologue=False)
    assert qasm == correct_qasm

def test_qasm_noprologue2():
    correct_qasm = "x q[0];\ny q[0];\nz q[0];"
    c = Circuit().x[0].y[0].z[0]
    qasm = c.to_qasm(False)
    assert qasm == correct_qasm

def test_from_qasm_roundtrip():
    c = from_qasm(QASM)
    assert c.to_qasm() == QASM

def test_from_qasm_angle_expressions():
    qasm = """rx(pi/2) q[0];
ry(-pi/4) q[1];
crx(pi) q[0],q[1];"""
    c = from_qasm(qasm)
    expected = Circuit(2).rx(math.pi / 2)[0].ry(-math.pi / 4)[1].crx(math.pi)[0, 1]
    assert np.allclose(c.run(), expected.run())

def test_from_qasm_multi_qubit_gates():
    qasm = """ccx q[0],q[1],q[2];
cswap q[0],q[1],q[2];
sdg q[3];
tdg q[3];
rzz(0.5) q[2],q[3];"""
    c = from_qasm(qasm)
    expected = Circuit(4).ccx[0, 1, 2].cswap[0, 1, 2].sdg[3].tdg[3].rzz(0.5)[2, 3]
    assert np.allclose(c.run(), expected.run())

def test_from_qasm_rejects_unsafe_expressions():
    with pytest.raises(ValueError):
        from_qasm('rx(__import__("os")) q[0];')

def test_from_qasm_rejects_unknown_gate():
    with pytest.raises(ValueError):
        from_qasm('bogusgate q[0];')

def test_barrier_qasm_roundtrip():
    # to_qasm emits real barrier statements; from_qasm must parse them back
    # (it used to skip them, so barriers vanished on a round trip).
    c = Circuit(3).h[0].barrier[:].cx[0, 1]
    c2 = from_qasm(c.to_qasm())
    assert [op.lowername for op in c2.ops] == ['h', 'barrier', 'cx']
    assert tuple(c2.ops[1].target_iter(3)) == (0, 1, 2)
    assert np.allclose(c.run(), c2.run())

def test_from_qasm_applies_a_whole_register_barrier():
    # "barrier q;" names the whole register. It used to be dropped, because the
    # parser tracked no register widths; now the declaration supplies them.
    c = from_qasm('qreg q[2]; h q[0]; barrier q; x q[0];')
    assert [op.lowername for op in c.ops] == ['h', 'barrier', 'x']
    assert set(c.ops[1].targets) == {0, 1}


def test_from_qasm_without_a_qreg_declaration_still_parses_indexed_targets():
    # No declaration to read a width from, so the width comes from the gates --
    # the old behaviour, kept for fragments that omit the header.
    c = from_qasm('h q[0]; x q[1];')
    assert [op.lowername for op in c.ops] == ['h', 'x']
    assert c.n_qubits == 2


def test_from_qasm_keeps_declared_but_idle_qubits():
    assert from_qasm('qreg q[5]; creg c[5]; h q[0];').n_qubits == 5


def test_from_qasm_round_trip_keeps_the_width():
    original = Circuit(4).h[0].cx[0, 1]
    assert from_qasm(original.to_qasm()).n_qubits == 4


def test_from_qasm_applies_a_gate_to_a_whole_register():
    c = from_qasm('qreg q[3]; h q;')
    assert [op.lowername for op in c.ops] == ['h', 'h', 'h']
    assert [op.targets for op in c.ops] == [0, 1, 2]


def test_from_qasm_measures_a_whole_register():
    c = from_qasm('qreg q[2]; creg c[2]; measure q -> c;')
    assert [op.lowername for op in c.ops] == ['measure', 'measure']


def test_from_qasm_rejects_an_unbounded_exponent():
    # The angle parser avoids eval, but ** is unbounded arithmetic and this text
    # can arrive from an MCP client; 9**9**9 never returns.
    with pytest.raises(ValueError, match='Exponent'):
        from_qasm('qreg q[1]; rx(9**9**9) q[0];')
    assert from_qasm('qreg q[1]; rx(2**3) q[0];').ops[0].theta == 8
