"""Tests for the MCP server tool functions and the safe Hamiltonian parser.

The tool implementations are plain functions, so most tests need no MCP
client; server construction itself is tested only when the optional `mcp`
dependency is installed.
"""
import math

import pytest
import torch

from blueqat import Circuit
from blueqat.mcp_server import (blueqat_info, circuit_stats, draw_circuit_png,
                                eo_transpile, expectation_value, run_circuit)
from blueqat.utils import X, Z, parse_hamiltonian

BELL = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
"""


# --- parse_hamiltonian ---------------------------------------------------------

def test_parse_hamiltonian_basic():
    h = parse_hamiltonian("1.5*Z[0]*Z[1] - 0.5*X[0] + 2")
    expected = (1.5 * Z[0] * Z[1] - 0.5 * X[0] + 2).simplify()
    assert h == expected


def test_parse_hamiltonian_index_without_brackets_and_no_star():
    assert parse_hamiltonian("2*Z0 Z1") == (2 * Z[0] * Z[1]).simplify()
    assert parse_hamiltonian("Z0Z1") == (Z[0] * Z[1]).to_expr().simplify()


def test_parse_hamiltonian_scientific_notation():
    h = parse_hamiltonian("1e-2*Z[0] + 1.5e+1*X[1]")
    assert h == (0.01 * Z[0] + 15.0 * X[1]).simplify()


def test_parse_hamiltonian_identity_and_sign():
    assert parse_hamiltonian("-Z[0]") == (-1.0 * Z[0]).to_expr().simplify()
    assert parse_hamiltonian("3") == (3 + 0 * Z[0]).simplify()
    assert parse_hamiltonian("2*I") == (2 + 0 * Z[0]).simplify()


@pytest.mark.parametrize("bad", ["", "  ", "Z", "Q[0]", "Z[0]$", "1..2*Z[0]"])
def test_parse_hamiltonian_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_hamiltonian(bad)


def test_parse_hamiltonian_no_eval():
    with pytest.raises(ValueError):
        parse_hamiltonian("__import__('os').system('true')")


# --- tools ----------------------------------------------------------------------

def test_run_circuit_statevector():
    out = run_circuit(BELL)
    assert out["n_qubits"] == 2
    sv = out["statevector"]
    assert sv[0][0] == pytest.approx(1 / math.sqrt(2))
    assert sv[3][0] == pytest.approx(1 / math.sqrt(2))
    assert sv[1] == [0.0, 0.0] and sv[2] == [0.0, 0.0]


def test_run_circuit_shots():
    out = run_circuit(BELL + "measure q[0] -> c[0];\nmeasure q[1] -> c[1];",
                      shots=200)
    assert sum(out["counts"].values()) == 200
    assert set(out["counts"]) <= {"00", "11"}


def test_run_circuit_wide_returns_top_probabilities():
    qasm = "\n".join(f"h q[{i}];" for i in range(12))
    out = run_circuit(qasm)
    assert "statevector" not in out
    assert len(out["top_probabilities"]) <= 20
    for p in out["top_probabilities"].values():
        assert p == pytest.approx(1 / 4096, rel=1e-6)


def test_run_circuit_validates_inputs():
    with pytest.raises(ValueError):
        run_circuit(BELL, backend="bogus")
    with pytest.raises(ValueError):
        run_circuit(BELL, shots=0)
    with pytest.raises(ValueError):
        run_circuit("bogusgate q[0];")


def test_circuit_stats():
    out = circuit_stats(BELL)
    assert out == {"n_qubits": 2, "depth": 2,
                   "gate_counts": {"h": 1, "cx": 1}}


def test_expectation_value():
    out = expectation_value("rx(0.4) q[0];", "Z[0]")
    assert out["expectation_value"] == pytest.approx(math.cos(0.4), abs=1e-7)


def test_draw_circuit_png():
    png = draw_circuit_png(BELL)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 1000


def test_eo_transpile():
    out = eo_transpile(BELL)
    assert out["logical_qubits"] == 2
    assert out["physical_spins"] == 6
    assert out["n_pulses"] == 31  # H = 3 pulses + FW-CNOT = 28
    assert out["parallel_speedup"] > 1.0
    assert len(out["pulses_preview"]) == 10


def test_blueqat_info():
    out = blueqat_info()
    assert "version" in out and "OpenQASM" in out["circuit_format"]


# --- server construction (only with the optional dependency) ---------------------

def test_build_server_registers_tools():
    pytest.importorskip("mcp")
    from blueqat.mcp_server import build_server
    server = build_server()
    import anyio
    tools = anyio.run(server.list_tools)
    names = {t.name for t in tools}
    assert {"run_circuit", "circuit_stats", "expectation_value",
            "draw_circuit", "eo_transpile", "blueqat_info"} <= names
