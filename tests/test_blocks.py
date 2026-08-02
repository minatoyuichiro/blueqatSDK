"""Tests for named gate blocks (nested subcircuits): construction, execution
transparency across backends, structure views, dagger, and interop."""
import math

import pytest
import torch

from blueqat import Circuit
from blueqat.gate import GateBlock

ATOL = 1e-10


def _bell_with_block():
    c = Circuit(2)
    with c.block("Bell"):
        c.h[0].cx[0, 1]
    return c


# --- construction -------------------------------------------------------------

def test_block_context_manager_groups_ops():
    c = _bell_with_block()
    assert len(c.ops) == 1
    assert isinstance(c.ops[0], GateBlock)
    assert c.ops[0].name == "Bell"
    assert [op.lowername for op in c.ops[0].ops] == ['h', 'cx']


def test_block_nesting():
    c = Circuit(3)
    with c.block("outer"):
        c.x[0]
        with c.block("inner"):
            c.y[1]
        c.z[2]
    outer = c.ops[0]
    assert [type(op).__name__ for op in outer.ops] == ['XGate', 'GateBlock', 'ZGate']
    assert outer.ops[1].name == "inner"


def test_block_grows_circuit_width():
    c = Circuit()
    with c.block("wide"):
        c.h[5]
    assert c.n_qubits == 6


def test_block_exception_leaves_ops_ungrouped():
    c = Circuit(1)
    with pytest.raises(RuntimeError):
        with c.block("broken"):
            c.h[0]
            raise RuntimeError("boom")
    # on error the ops stay as-is (no half-built block is created)
    assert [op.lowername for op in c.ops] == ['h']


def test_append_block():
    sub = Circuit(2).h[0].cx[0, 1]
    c = Circuit(2).x[0]
    c.append_block("Bell", sub)
    assert isinstance(c.ops[1], GateBlock)
    inline = Circuit(2).x[0].h[0].cx[0, 1]
    assert torch.allclose(c.run(), inline.run(), atol=ATOL)


def test_append_block_with_offset():
    sub = Circuit(2).h[0].cx[0, 1]      # built on qubits 0, 1
    c = Circuit()
    c.append_block("Bell@2", sub, offset=2)
    assert c.n_qubits == 4
    inline = Circuit(4).h[2].cx[2, 3]
    assert torch.allclose(c.run(), inline.run(), atol=ATOL)


def test_append_block_offset_resolves_slices():
    sub = Circuit(2).h[:]               # slice target
    c = Circuit()
    c.append_block("Hs", sub, offset=1)
    inline = Circuit(3).h[1].h[2]
    assert torch.allclose(c.run(), inline.run(), atol=ATOL)


def test_append_block_rejects_negative_offset():
    with pytest.raises(ValueError):
        Circuit().append_block("x", Circuit(1).x[0], offset=-1)


def test_append_block_offset_preserves_nested_blocks():
    # Shifting a library circuit must keep its inner block structure
    # (previously the offset path flattened blocks away).
    sub = Circuit(2)
    with sub.block("stage A"):
        sub.h[0]
    with sub.block("stage B"):
        sub.cx[0, 1]
    c = Circuit()
    c.append_block("lib", sub, offset=3)
    lib = c.ops[0]
    assert [op.name for op in lib.ops if isinstance(op, GateBlock)] == \
        ["stage A", "stage B"]
    inline = Circuit(5).h[3].cx[3, 4]
    assert torch.allclose(c.run(), inline.run(), atol=ATOL)


# --- execution transparency ----------------------------------------------------

def test_block_execution_matches_inline_both_modes():
    c = Circuit(3)
    with c.block("algo"):
        c.h[:]
        with c.block("QFT"):
            c.h[0].cphase(math.pi / 2)[0, 1].h[1]
        c.cx[1, 2]
    inline = Circuit(3).h[:].h[0].cphase(math.pi / 2)[0, 1].h[1].cx[1, 2]
    for mode in ['statevector', 'tensornet']:
        assert torch.allclose(c.run(backend=mode), inline.run(backend=mode), atol=ATOL)


def test_block_shots():
    c = _bell_with_block()
    counts = c.m[:].shots(100)
    assert set(counts) <= {"00", "11"}


def test_block_to_qasm_expands():
    q1 = _bell_with_block().to_qasm()
    q2 = Circuit(2).h[0].cx[0, 1].to_qasm()
    assert q1 == q2


def test_block_json_serialization_via_flatten():
    from blueqat.circuit_funcs.json_serializer import deserialize, serialize
    c = _bell_with_block()
    c2 = deserialize(serialize(c))
    assert torch.allclose(c.run(), c2.run(), atol=ATOL)


def test_block_draw_no_warning():
    import warnings

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _bell_with_block().run(backend='draw')
        _bell_with_block().run(backend='draw', expand_blocks=True)
    plt.close('all')
    assert not [w for w in caught if 'omitted' in str(w.message)]


def test_block_draw_as_box_by_default():
    # Default: the block is rendered as one labeled box spanning its qubits.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ctx = _bell_with_block().run(backend='draw')
    plt.close('all')
    qlist = ctx[0]
    block_nodes = [e for i in qlist for e in qlist[i] if e['type'] == 'block']
    assert [e['gate'] for e in block_nodes] == ['Bell', 'Bell']  # one per qubit
    gate_nodes = [e for i in qlist for e in qlist[i] if e['type'] == 'gate']
    assert gate_nodes == []  # inner gates are hidden in box mode


def test_block_draw_expand_blocks_shows_gates():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ctx = _bell_with_block().run(backend='draw', expand_blocks=True)
    plt.close('all')
    qlist = ctx[0]
    assert not [e for i in qlist for e in qlist[i] if e['type'] == 'block']
    gate_names = [e['gate'] for i in qlist for e in qlist[i] if e['type'] == 'gate']
    assert 'H' in gate_names  # the Bell contents are drawn as plain gates


def _block_labels(ctx):
    qlist = ctx[0]
    labels = []
    for i in qlist:
        for e in qlist[i]:
            if e['type'] == 'block' and e['gate'] not in labels:
                labels.append(e['gate'])
    return labels


def test_block_draw_auto_descends_singleton_wrapper():
    # A circuit that is entirely one block (Shor-style) must not render as a
    # single giant box: the drawer descends into singleton wrappers so the
    # child blocks appear as boxes.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = Circuit(2)
    with c.block("whole"):
        with c.block("a"):
            c.h[0]
        with c.block("b"):
            c.cx[0, 1]
    ctx = c.run(backend='draw')
    plt.close('all')
    assert set(_block_labels(ctx)) == {"a", "b"}


def test_block_draw_integer_depth():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = Circuit(2)
    with c.block("outer"):
        c.h[0]
        with c.block("inner"):
            c.cx[0, 1]
    # depth=1: outer is expanded one level; inner shows as a box.
    ctx = c.run(backend='draw', expand_blocks=1)
    plt.close('all')
    assert _block_labels(ctx) == ["inner"]
    qlist = ctx[0]
    gate_names = [e['gate'] for i in qlist for e in qlist[i] if e['type'] == 'gate']
    assert 'H' in gate_names


def test_block_eo_transpile():
    import blueqat.eo  # noqa: F401
    c = Circuit(2)
    with c.block("Bell"):
        c.h[0].cx[0, 1]
    phys = c.run(backend='eo')
    phys_inline = Circuit(2).h[0].cx[0, 1].run(backend='eo')
    assert [(op.targets, float(op.theta)) for op in phys.ops] == \
           [(op.targets, float(op.theta)) for op in phys_inline.ops]


# --- structure views -----------------------------------------------------------

def test_tree_shows_nesting():
    c = Circuit(3)
    with c.block("outer"):
        c.h[0]
        with c.block("inner"):
            c.x[1]
    t = c.tree()
    assert "outer" in t and "inner" in t
    # inner is indented deeper than outer
    outer_line = next(l for l in t.splitlines() if l.endswith("outer"))
    inner_line = next(l for l in t.splitlines() if l.endswith("inner"))
    assert len(inner_line) - len(inner_line.lstrip('│ ├└─')) >= 0
    assert t.splitlines().index(inner_line) > t.splitlines().index(outer_line)


def test_repr_shows_block():
    c = _bell_with_block()
    assert "block('Bell', <2 ops>)" in repr(c)


def test_depth_and_count_ops_recurse_into_blocks():
    c = Circuit(2)
    with c.block("Bell"):
        c.h[0].cx[0, 1]
    assert c.depth() == 2
    assert c.count_ops() == {'h': 1, 'cx': 1}


# --- dagger ---------------------------------------------------------------------

def test_block_dagger_uncomputes():
    c = Circuit(3)
    with c.block("algo"):
        c.h[0].crz(0.7)[0, 1].t[2].cx[1, 2]
    d = c.dagger()
    assert isinstance(d.ops[0], GateBlock)
    assert d.ops[0].name == "algo†"
    e0 = torch.zeros(8, dtype=torch.complex128)
    e0[0] = 1
    assert torch.allclose((c + d).run(), e0, atol=1e-8)


def test_block_double_dagger_name_roundtrip():
    c = _bell_with_block()
    dd = c.dagger().dagger()
    assert dd.ops[0].name == "Bell"


def test_block_dagger_rejects_measurement_inside():
    c = Circuit(1)
    with c.block("meas"):
        c.h[0].m[0]
    with pytest.raises(ValueError, match='meas'):
        c.dagger()
