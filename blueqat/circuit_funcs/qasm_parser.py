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
"""Parser for a practical subset of OpenQASM 2.0 (the qelib1.inc gate set) into a Circuit.

This is the reverse of `Circuit.to_qasm()`.
"""

import ast
import math
import operator
import re
from typing import Dict, List, Optional, Tuple

from ..circuit import Circuit

# name -> (blueqat gate name, number of numeric args)
_GATES = {
    # no-arg 1-qubit
    'x': ('x', 0), 'y': ('y', 0), 'z': ('z', 0), 'h': ('h', 0),
    's': ('s', 0), 'sdg': ('sdg', 0), 't': ('t', 0), 'tdg': ('tdg', 0),
    'sx': ('sx', 0), 'sxdg': ('sxdg', 0), 'id': ('i', 0), 'reset': ('reset', 0),
    # 1-arg 1-qubit
    'rx': ('rx', 1), 'ry': ('ry', 1), 'rz': ('rz', 1),
    'p': ('phase', 1), 'u1': ('phase', 1),
    # 3-arg 1-qubit
    'u': ('u', 3), 'u3': ('u', 3),
    # no-arg 2-qubit
    'cx': ('cx', 0), 'cz': ('cz', 0), 'cy': ('cy', 0), 'ch': ('ch', 0), 'swap': ('swap', 0),
    # 1-arg 2-qubit
    'cp': ('cphase', 1), 'cu1': ('cphase', 1),
    'crx': ('crx', 1), 'cry': ('cry', 1), 'crz': ('crz', 1),
    'rxx': ('rxx', 1), 'ryy': ('ryy', 1), 'rzz': ('rzz', 1),
    # blueqat's zz gate is the fixed diag(1, i, i, 1) -- it takes no angle
    'zz': ('zz', 0),
    # 4-arg 2-qubit
    'cu': ('cu', 4),
    # no-arg 3-qubit
    'ccx': ('ccx', 0), 'cswap': ('cswap', 0),
}

_CONSTS = {'pi': math.pi}
#: Largest exponent an angle expression may use. `**` is unbounded arithmetic:
#: `9**9**9` has no answer a machine will finish computing, and this parser reads
#: text arriving from MCP clients, so an angle is not a place to allow that.
MAX_EXPONENT = 64


def _bounded_pow(base: float, exponent: float) -> float:
    if abs(exponent) > MAX_EXPONENT:
        raise ValueError(
            f"Exponent {exponent} exceeds the limit of {MAX_EXPONENT} allowed in a "
            f"QASM angle expression.")
    return operator.pow(base, exponent)


_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: _bounded_pow}
_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval_angle_expr(node: ast.AST) -> float:
    """Safely evaluate a numeric angle expression (numbers, +-*/, unary -, and `pi`)
    without falling back to `eval`, since this parses (potentially untrusted) QASM text."""
    if isinstance(node, ast.Expression):
        return _eval_angle_expr(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_angle_expr(node.left), _eval_angle_expr(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_angle_expr(node.operand))
    raise ValueError(f"Unsupported expression in QASM angle argument: {ast.dump(node)}")


def _parse_args(args_str: str) -> List[float]:
    args = []
    for a in args_str.split(','):
        try:
            tree = ast.parse(a.strip(), mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Could not parse QASM argument {a!r}: {e}") from e
        args.append(_eval_angle_expr(tree))
    return args


def _arity(gate_name: str) -> int:
    """How many qubits one application of this gate takes."""
    from ..gate import OneQubitGate, TwoQubitGate
    from ..gateset import get_op_type
    op_type = get_op_type(gate_name)
    if op_type is None or issubclass(op_type, OneQubitGate):
        return 1
    if issubclass(op_type, TwoQubitGate):
        return 2
    return 3


def _parse_registers(text: str) -> Tuple[Dict[str, Tuple[int, int]], int]:
    """Declared quantum registers as ``name -> (offset, size)``, and the total width.

    Registers are laid out consecutively in blueqat's single qubit space, in
    declaration order.
    """
    registers: Dict[str, Tuple[int, int]] = {}
    offset = 0
    for name, size in re.findall(r'\bqreg\s+(\w+)\s*\[\s*(\d+)\s*\]', text):
        size = int(size)
        registers[name] = (offset, size)
        offset += size
    return registers, offset


def _parse_targets(targets_str: str,
                   registers: Optional[Dict[str, Tuple[int, int]]] = None) -> List[int]:
    """Qubit indices named by a target list.

    ``q[2]`` names one qubit; a bare ``q`` names the whole register, which is
    ordinary OpenQASM and used to parse as an empty target list -- so the gate
    was appended and did nothing, with no error.
    """
    registers = registers or {}
    targets: List[int] = []
    for piece in targets_str.split(','):
        piece = piece.strip()
        if not piece:
            continue
        indexed = re.fullmatch(r'(\w+)\s*\[\s*(\d+)\s*\]', piece)
        if indexed:
            name, index = indexed.group(1), int(indexed.group(2))
            offset = registers.get(name, (0, 0))[0] if registers else 0
            targets.append(offset + index)
            continue
        whole = re.fullmatch(r'\w+', piece)
        if whole and whole.group(0) in registers:
            offset, size = registers[whole.group(0)]
            targets.extend(range(offset, offset + size))
            continue
        raise ValueError(f"Could not parse QASM target {piece!r}.")
    return targets


def look_alike_characters(text: str) -> Dict[str, str]:
    """Characters that are not what they look like, and what they should be.

    Text pasted out of a PDF or a word processor carries characters that render
    identically to ASCII or to an ordinary ideograph but are different code
    points -- full-width punctuation and digits, and the 214 Kangxi radicals,
    where ``子`` (U+5B50) and ``⼦`` (U+2F26) are indistinguishable on screen. A
    program carrying them fails to parse for a reason nobody can see by looking.

    Returns ``{character: what it normalizes to}`` for each distinct offender.
    """
    import unicodedata
    out: Dict[str, str] = {}
    for ch in text:
        if ord(ch) < 128:
            continue
        replacement = unicodedata.normalize('NFKC', ch)
        if replacement != ch:
            out[ch] = replacement
    return out


def _look_alike_note(text: str) -> str:
    """A sentence naming the look-alikes in `text`, or nothing."""
    offenders = look_alike_characters(text)
    if not offenders:
        return ""
    shown = ', '.join(f"{ch!r} (U+{ord(ch):04X}) for {want!r}"
                      for ch, want in list(offenders.items())[:4])
    more = '' if len(offenders) <= 4 else f", and {len(offenders) - 4} more"
    return (f" The program contains characters that look like ASCII but are "
            f"not: {shown}{more}. Text pasted from a PDF or a word processor "
            f"does this. Pass normalize=True to from_qasm, or run the source "
            f"through unicodedata.normalize('NFKC', ...) first.")


def from_qasm(qasm: str, normalize: bool = False) -> Circuit:
    """Parse an OpenQASM 2.0 program (the qelib1.inc gate set) into a Circuit.

    `normalize` applies NFKC first, folding full-width punctuation and Kangxi
    radicals onto their ASCII and ideographic equivalents. It is off by default
    because silently rewriting input is a guess at what was meant; when parsing
    fails, the error says whether such characters are present, which is
    something no amount of looking at the text will reveal.
    """
    import unicodedata
    if normalize:
        qasm = unicodedata.normalize('NFKC', qasm)
    try:
        return _parse_program(qasm)
    except ValueError as e:
        note = _look_alike_note(qasm)
        raise ValueError(f"{e}{note}") from None


def _parse_program(qasm: str) -> Circuit:
    # strip line (//) and block (/* */) comments
    text = re.sub(r'/\*.*?\*/', '', qasm, flags=re.DOTALL)
    text = re.sub(r'//.*', '', text)

    # The declared width is the circuit's width. Inferring it from the gates
    # instead loses every qubit that was declared but left idle, and shrinks a
    # circuit on a to_qasm/from_qasm round trip.
    registers, width = _parse_registers(text)
    c = Circuit(width)

    for raw_stmt in text.split(';'):
        stmt = raw_stmt.strip()
        if not stmt:
            continue
        if stmt.startswith(('OPENQASM', 'include', 'qreg', 'creg', 'gate ', 'opaque ')):
            continue

        barrier_match = re.match(r'barrier\s+(.+)$', stmt)
        if barrier_match:
            targets = _parse_targets(barrier_match.group(1), registers)
            if targets:
                c.barrier[tuple(targets) if len(targets) > 1 else targets[0]]
            continue

        measure_match = re.match(r'measure\s+(.+?)\s*->\s*(.+)$', stmt)
        if measure_match:
            targets = _parse_targets(measure_match.group(1), registers)
            for target in targets:
                c.m[target]
            continue

        gate_match = re.match(r'(\w+)\s*(?:\(([^)]*)\))?\s+(.+)$', stmt)
        if not gate_match:
            raise ValueError(f"Could not parse QASM statement: {stmt!r}")
        name, args_str, targets_str = gate_match.groups()

        if name not in _GATES:
            raise ValueError(f"Unsupported QASM gate: {name!r}")
        gate_name, n_args = _GATES[name]
        args = _parse_args(args_str) if args_str else []
        if len(args) != n_args:
            raise ValueError(f"Gate {name!r} expects {n_args} argument(s), got {len(args)}")
        targets = _parse_targets(targets_str, registers)
        if not targets:
            raise ValueError(f"Gate {name!r} names no qubits: {stmt!r}")

        op = getattr(c, gate_name)
        if args:
            op = op(*args)
        if _arity(gate_name) == 1 and len(targets) > 1:
            # "h q;" over a whole register is one gate per qubit.
            for target in targets:
                op[target]
        else:
            op[targets[0] if len(targets) == 1 else tuple(targets)]

    return c
