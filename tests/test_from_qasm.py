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


# --- which look-alikes are actually wrong ----------------------------------
#
# "Does NFKC change this?" is not the same question as "is this damaged?".
# NFKC turns ① into 1, ㎡ into m2, Ⅳ into IV and … into ..., none of which is a
# repair, and full-width brackets are correct Japanese typography. Using the
# wide test as a damage check reports good prose as broken, and normalizing on
# that basis destroys meaning rather than restoring it.

def test_correct_japanese_prose_has_nothing_always_wrong_in_it():
    from blueqat.circuit_funcs.qasm_parser import always_wrong_characters
    prose = '量子ビット（qubit）は①の条件で150㎡、Ⅳ章…'
    assert always_wrong_characters(prose) == {}
    # ...while the wide test flags every one of those, which is the point.
    assert len(look_alike_characters(prose)) == 6


@pytest.mark.parametrize('text,expected', [
    ('量⼦', {'⼦': '子'}),                          # Kangxi radical U+2F26
    ('ＦＡＱ', {'Ｆ': 'F', 'Ａ': 'A', 'Ｑ': 'Q'}),    # full-width letters
    ('２０２６', {'２': '2', '０': '0', '６': '6'}),   # full-width digits
])
def test_always_wrong_characters_catches_the_real_impostors(text, expected):
    from blueqat.circuit_funcs.qasm_parser import always_wrong_characters
    assert always_wrong_characters(text) == expected


def test_the_cjk_radicals_supplement_is_not_included():
    """It looks like it belongs and does not: 2 of its 115 characters change
    under NFKC, so listing it means reporting damage this cannot repair.

    An earlier version did include it, and this test asserted only that the
    result was non-empty -- which it was, because the character mapped to
    *itself*. It passed while the function named a problem and offered the
    problem as its own solution. Built from code points, never from a literal:
    writing the same character twice and comparing always agrees."""
    from blueqat.circuit_funcs.qasm_parser import always_wrong_characters
    supplement = chr(0x2E80)                  # CJK RADICAL REPEAT
    assert always_wrong_characters(supplement) == {}


def test_the_ranges_are_the_ones_nfkc_can_actually_repair():
    """Membership was decided by counting, not by which block a character
    looks like it belongs to."""
    import unicodedata
    from blueqat.circuit_funcs.qasm_parser import LOOK_ALIKE_RANGES
    for low, high in LOOK_ALIKE_RANGES:
        defined = [chr(cp) for cp in range(low, high + 1)
                   if unicodedata.name(chr(cp), None)]
        repaired = [c for c in defined if unicodedata.normalize('NFKC', c) != c]
        assert len(repaired) / len(defined) > 0.9, f"U+{low:04X}-{high:04X}"


def test_extraction_damage_is_separated_from_deliberate_typography():
    """The two want opposite treatment: damage can be repaired in the stored
    text, because nobody chose it; a full-width company name cannot, because
    somebody did."""
    from blueqat.circuit_funcs.qasm_parser import (extraction_damage,
                                                   presentation_variants)
    company = '博報堂' + chr(0xFF24) + chr(0xFF39)      # full-width D, Y
    assert extraction_damage(company) == {}
    assert set(presentation_variants(company)) == {chr(0xFF24), chr(0xFF39)}

    damaged = '量' + chr(0x2F26)                        # Kangxi radical child
    assert extraction_damage(damaged) == {chr(0x2F26): '子'}
    assert presentation_variants(damaged) == {}


def test_half_width_kana_is_a_presentation_choice_not_damage():
    from blueqat.circuit_funcs.qasm_parser import (extraction_damage,
                                                   presentation_variants)
    kana = chr(0xFF83) + chr(0xFF7D) + chr(0xFF84)       # half-width te su to
    assert extraction_damage(kana) == {}
    assert len(presentation_variants(kana)) == 3


def test_the_variants_normalizing_cannot_fix_are_reported_separately():
    """U+FA11 and U+5D0E stay different after NFKC on both sides, so a name
    written with one will not match the other. Saying a normalization pass
    resolved it is how the same report comes back a second time."""
    import unicodedata
    from blueqat.circuit_funcs.qasm_parser import (always_wrong_characters,
                                                   unfixable_lookalikes)
    variant, ordinary = chr(0xFA11), chr(0x5D0E)
    assert unicodedata.normalize('NFKC', variant) != unicodedata.normalize('NFKC', ordinary)
    assert always_wrong_characters(variant) == {}
    assert unfixable_lookalikes(variant) == {variant: variant}


def test_qasm_itself_is_stricter_than_prose():
    """A program is ASCII by definition, so from_qasm reports every look-alike
    rather than only the ones that are wrong anywhere."""
    with pytest.raises(ValueError, match='look like ASCII but are not'):
        from_qasm(QASM_HEAD + 'rx（0.5） q[0];')    # full-width brackets only
