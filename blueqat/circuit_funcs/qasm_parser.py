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

    Returns ``{character: what it normalizes to}`` for each distinct
    character NFKC would change.

    ⚠ That is a wide net, and using it as a "this text is damaged" test gives
    false positives on perfectly good Japanese. Full-width brackets and colons
    are correct typography; NFKC also turns ``①`` into ``1``, ``㎡`` into
    ``m2``, ``Ⅳ`` into ``IV`` and ``…`` into ``...``, none of which is a
    repair. Use `always_wrong_characters` to ask whether something is broken;
    use this one to describe what is there.
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


#: Look-alikes that are damage from extracting text, never an authored choice.
#: A reader seeing one is already looking at something broken, so these are the
#: ones it is safe to repair in the stored text itself. Counted over each
#: range, the share NFKC repairs:
#:
#:     Kangxi radicals          U+2F00-2FD5   214 of 214   100%
#:     CJK compatibility ideos  U+F900-FAFF   460 of 472  97.5%
#:
#: The CJK radicals supplement, U+2E80-2EF3, looks like it belongs here and
#: does not: 2 of its 115 characters change under NFKC, so including it means
#: reporting things this cannot repair.
EXTRACTION_DAMAGE_RANGES = (
    (0x2F00, 0x2FD5),      # Kangxi radicals
    (0xF900, 0xFAFF),      # CJK compatibility ideographs
)

#: Look-alikes that may be exactly what someone meant to write. Full-width
#: letters are the registered form of some company names and appear in
#: quotations that must not be altered; half-width kana is a presentation
#: choice. Normalize these into a *search index*, never in the stored text.
PRESENTATION_RANGES = (
    (0xFF10, 0xFF19),      # full-width 0-9
    (0xFF21, 0xFF3A),      # full-width A-Z
    (0xFF41, 0xFF5A),      # full-width a-z
    (0xFF61, 0xFF9F),      # half-width kana
)

#: Both, which is what a *program* cares about: QASM is ASCII by definition, so
#: the distinction between damage and intent does not arise inside one.
LOOK_ALIKE_RANGES = EXTRACTION_DAMAGE_RANGES + PRESENTATION_RANGES


def _in_ranges(text: str, ranges, repairable: bool = True) -> Dict[str, str]:
    import unicodedata
    out: Dict[str, str] = {}
    for ch in text:
        if not any(low <= ord(ch) <= high for low, high in ranges):
            continue
        replacement = unicodedata.normalize('NFKC', ch)
        if (replacement != ch) == repairable:
            out[ch] = replacement
    return out


def extraction_damage(text: str) -> Dict[str, str]:
    """Look-alikes that got there by accident, and their repair.

    Kangxi radicals and compatibility ideographs come from extracting text, not
    from anyone typing them: U+2F26 renders exactly like U+5B50 and is a
    different character, so a document carrying them cannot be searched for its
    own words. Because nobody chose them, these are safe to fix in the stored
    text -- quoting and display get better too.
    """
    return _in_ranges(text, EXTRACTION_DAMAGE_RANGES)


def presentation_variants(text: str) -> Dict[str, str]:
    """Look-alikes that may be deliberate. Normalize an index, not the text.

    Full-width letters are the registered form of some company names and appear
    inside quotations that must stay as they were; half-width kana is a
    presentation choice. Nothing here can tell an intentional one from an
    accident, so rewriting the stored text changes names and misquotes sources.
    Normalize both the index and the query instead, and leave the text alone.
    """
    return _in_ranges(text, PRESENTATION_RANGES)


def always_wrong_characters(text: str) -> Dict[str, str]:
    """Every look-alike, damage or presentation, with its repair.

    The union of `extraction_damage` and `presentation_variants`. Right for a
    program -- QASM is ASCII, so anything here is a mistake in one -- and the
    wrong question for prose, where the two halves want opposite treatment.

    Only characters NFKC actually changes are returned. Reporting one whose
    normalized form is itself would be naming a problem and offering the
    problem as its own solution; see `unfixable_lookalikes`.
    """
    return _in_ranges(text, LOOK_ALIKE_RANGES)


def unfixable_lookalikes(text: str) -> Dict[str, str]:
    """Look-alikes that normalizing will *not* resolve.

    Twelve compatibility ideographs -- U+FA0E, FA0F, FA11, FA13, FA14, FA1F,
    FA21, FA23, FA24, FA27, FA28 and FA29 -- normalize to themselves, so U+FA11
    and U+5D0E stay different after NFKC on both sides. Variant forms of a
    personal name are the usual way to meet them, and they need a different
    answer entirely.

    Returned as ``{character: itself}`` so that "found it" and "fixed it" stay
    distinguishable: calling a normalization pass a resolution here is how the
    same report comes back a second time.
    """
    return _in_ranges(text, LOOK_ALIKE_RANGES, repairable=False)


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
