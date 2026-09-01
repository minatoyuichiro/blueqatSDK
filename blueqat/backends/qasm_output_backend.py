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

from collections import deque

from ..gate import *
from .backendbase import Backend


def _assign_classical_bits(gates, n_qubits):
    """Which classical bit each measurement writes to, and how wide `creg` must be.

    The first measurement of a qubit keeps ``c[qubit]``, so a circuit that
    measures each qubit at most once emits exactly the QASM it always did. A
    later measurement of the same qubit takes a fresh bit past the quantum
    register's width instead: writing it to ``c[qubit]`` again would overwrite
    the earlier result, which is precisely what a circuit that reuses a qubit
    measured it for.

    A measurement's `key` names a result, so measurements sharing one write to
    the same classical bit -- which is what makes ``m(key="a")`` mean the same
    thing in the exported QASM as it does when the circuit is simulated. The
    exception is ``duplicated="append"``, where blueqat collects a list of
    separate results and each therefore needs a bit of its own.

    Returns the per-measurement assignments in the order they will be emitted,
    each with a note naming the qubit and key, since OpenQASM 2.0 has nowhere
    else to record them.
    """
    assignments = []
    used_bits = set()
    bit_of_key: dict = {}
    next_free = n_qubits
    for gate in gates:
        if gate.lowername != 'measure':
            continue
        key = getattr(gate, 'key', None)
        appending = getattr(gate, 'duplicated', None) == 'append'
        for qubit in gate.target_iter(n_qubits):
            slot = (key, qubit)
            if key is not None and not appending and slot in bit_of_key:
                bit = bit_of_key[slot]
            elif qubit not in used_bits:
                bit = qubit
            else:
                bit = next_free
                next_free += 1
            used_bits.add(bit)
            if key is not None and not appending:
                bit_of_key[slot] = bit
            note = ''
            if bit != qubit or key is not None:
                note = f'q[{qubit}]'
                if key is not None:
                    note += f' key "{key}"'
            assignments.append((bit, note))
    return assignments, max(n_qubits, next_free)


class QasmOutputBackend(Backend):
    """Backend for OpenQASM output."""
    def _preprocess_run(self, gates, n_qubits, args, kwargs):
        def _parse_run_args(output_prologue=True, **_kwargs):
            return {'output_prologue': output_prologue}

        args = _parse_run_args(*args, **kwargs)
        assignments, creg_width = _assign_classical_bits(gates, n_qubits)
        if args['output_prologue']:
            qasmlist = [
                "OPENQASM 2.0;",
                'include "qelib1.inc";',
                f"qreg q[{n_qubits}];",
                f"creg c[{creg_width}];",
            ]
        else:
            qasmlist = []
        return gates, (qasmlist, n_qubits, deque(assignments))

    def _postprocess_run(self, ctx):
        return "\n".join(ctx[0])

    def _one_qubit_gate_noargs(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append(f"{gate.lowername} q[{idx}];")
        return ctx

    gate_x = _one_qubit_gate_noargs
    gate_y = _one_qubit_gate_noargs
    gate_z = _one_qubit_gate_noargs
    gate_h = _one_qubit_gate_noargs
    gate_t = _one_qubit_gate_noargs
    gate_s = _one_qubit_gate_noargs
    gate_sx = _one_qubit_gate_noargs
    gate_sxdg = _one_qubit_gate_noargs

    def _two_qubit_gate_noargs(self, gate, ctx):
        for control, target in gate.control_target_iter(ctx[1]):
            ctx[0].append(f"{gate.lowername} q[{control}],q[{target}];")
        return ctx

    gate_cz = _two_qubit_gate_noargs
    gate_cx = _two_qubit_gate_noargs
    gate_cy = _two_qubit_gate_noargs
    gate_ch = _two_qubit_gate_noargs
    gate_swap = _two_qubit_gate_noargs

    def _one_qubit_gate_args_theta(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append(f"{gate.lowername}({gate.theta}) q[{idx}];")
        return ctx

    gate_rx = _one_qubit_gate_args_theta
    gate_ry = _one_qubit_gate_args_theta
    gate_rz = _one_qubit_gate_args_theta

    def gate_i(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append(f"id q[{idx}];")
        return ctx

    def gate_u(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            if abs(gate.gamma) > 1e-7:
                ctx[0].append(
                    f"{gate.lowername}({gate.theta},{gate.phi},{gate.lam}) q[{idx}]; // global phase e^i{gate.gamma} is ignored."
                )
            else:
                ctx[0].append(
                    f"{gate.lowername}({gate.theta},{gate.phi},{gate.lam}) q[{idx}];"
                )
        return ctx

    def gate_phase(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            ctx[0].append(
                f"p({gate.theta}) q[{idx}];")
        return ctx

    def gate_cu(self, gate, ctx):
        for c, t in gate.control_target_iter(ctx[1]):
            ctx[0].append(
                f"{gate.lowername}({gate.theta},{gate.phi},{gate.lam},{gate.gamma}) q[{c}],q[{t}];"
            )
        return ctx

    def gate_cphase(self, gate, ctx):
        for c, t in gate.control_target_iter(ctx[1]):
            ctx[0].append(f"cp({gate.theta}) q[{c}],q[{t}];")
        return ctx

    def _two_qubit_gate_args_theta(self, gate, ctx):
        for c, t in gate.control_target_iter(ctx[1]):
            ctx[0].append(f"{gate.lowername}({gate.theta}) q[{c}],q[{t}];")
        return ctx

    gate_crx = _two_qubit_gate_args_theta
    gate_cry = _two_qubit_gate_args_theta
    gate_crz = _two_qubit_gate_args_theta
    gate_rxx = _two_qubit_gate_args_theta
    gate_ryy = _two_qubit_gate_args_theta
    gate_rzz = _two_qubit_gate_args_theta

    def gate_zz(self, gate, ctx):
        # ZZ = diag(1, i, i, 1) = e^{i pi/4} * rzz(pi/2); QASM 2.0 can't express
        # the global phase, so it's noted and dropped (same policy as gate_u).
        for c, t in gate.control_target_iter(ctx[1]):
            ctx[0].append(f"rzz(pi/2) q[{c}],q[{t}]; // global phase e^(i*pi/4) is ignored.")
        return ctx

    def gate_zzdg(self, gate, ctx):
        for c, t in gate.control_target_iter(ctx[1]):
            ctx[0].append(f"rzz(-pi/2) q[{c}],q[{t}]; // global phase e^(-i*pi/4) is ignored.")
        return ctx

    def _three_qubit_gate_noargs(self, gate, ctx):
        c0, c1, t = gate.targets
        ctx[0].append(f"{gate.lowername} q[{c0}],q[{c1}],q[{t}];")
        return ctx

    gate_ccx = _three_qubit_gate_noargs
    gate_cswap = _three_qubit_gate_noargs

    def gate_measure(self, gate, ctx):
        for idx in gate.target_iter(ctx[1]):
            bit, note = ctx[2].popleft()
            comment = f"  // {note}" if note else ""
            ctx[0].append(f"measure q[{idx}] -> c[{bit}];{comment}")
        return ctx

    gate_reset = _one_qubit_gate_noargs

    def gate_barrier(self, gate, ctx):
        qubits = ",".join(f"q[{idx}]" for idx in gate.target_iter(ctx[1]))
        ctx[0].append(f"barrier {qubits};")
        return ctx
