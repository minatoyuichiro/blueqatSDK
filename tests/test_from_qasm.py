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
"""Reading OpenQASM that came from somewhere else."""

import pytest

from blueqat.circuit_funcs.qasm_parser import from_qasm, look_alike_characters


# --- characters that are not what they look like ---------------------------
#
# Text pasted out of a PDF or a word processor carries full-width punctuation
# and digits, and the 214 Kangxi radicals: 子 (U+5B50) and ⼦ (U+2F26) are
# indistinguishable on screen and are different code points. A program carrying
# them fails to parse for a reason nobody can see by looking at it.

QASM_HEAD = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n'


@pytest.mark.parametrize('body', [
    'ｒｘ(0.5) q[0];',        # full-width letters
    'rx(０.５) q[0];',        # full-width digits
    'rx（0.5） q[0];',        # full-width parentheses
    'rx(0.5) q［0］;',        # full-width brackets
])
def test_lookalike_characters_are_named_in_the_error(body):
    with pytest.raises(ValueError) as exc:
        from_qasm(QASM_HEAD + body)
    message = str(exc.value)
    assert 'look like ASCII but are not' in message
    assert 'U+FF' in message              # the actual code point, not a guess
    assert 'normalize=True' in message


def test_an_ordinary_error_is_not_decorated():
    """The note appears only when there is something to note."""
    with pytest.raises(ValueError) as exc:
        from_qasm(QASM_HEAD + 'foo(0.5) q[0];')
    assert 'look like ASCII' not in str(exc.value)


def test_normalizing_is_opt_in_and_then_it_parses():
    """Off by default: silently rewriting the input is a guess at what was
    meant, and the error already says what to do."""
    mangled = QASM_HEAD + 'ｒｘ（０.５） q［0］;'
    with pytest.raises(ValueError):
        from_qasm(mangled)
    circuit = from_qasm(mangled, normalize=True)
    assert len(circuit.ops) == 1
    assert circuit.ops[0].lowername == 'rx'


def test_kangxi_radicals_are_reported_too():
    """All 214 of them look like ordinary ideographs. They cannot appear in
    valid QASM, but they can appear in a comment or an identifier pasted along
    with it, and then nothing about the failure is visible."""
    found = look_alike_characters('量⼦')          # second character is U+2F26
    assert found == {'⼦': '子'}


def test_japanese_prose_is_not_flagged_as_broken():
    """Full-width punctuation is correct Japanese typography, not damage --
    but NFKC does change it, so a bare "does NFKC alter this?" check reports
    ordinary Japanese text as needing repair."""
    found = look_alike_characters('量子ビット（qubit）')
    assert set(found) == {'（', '）'}            # only the punctuation
    assert '量' not in found and '子' not in found
