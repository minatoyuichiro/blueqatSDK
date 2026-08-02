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
"""MCP (Model Context Protocol) server exposing blueqat to LLM clients.

Install the optional dependency and register the server with an MCP client
(Claude Desktop, Claude Code, ...):

    pip install blueqat[mcp]

    // e.g. Claude Desktop's config:
    { "mcpServers": { "blueqat": { "command": "blueqat-mcp" } } }

Circuits are exchanged as OpenQASM 2.0 text (parsed with blueqat's eval-free
parser) and Hamiltonians as Pauli-expression strings parsed by
:func:`blueqat.utils.parse_hamiltonian` -- no code execution ever happens on
tool inputs.

The tool implementations below are plain functions returning JSON-compatible
dicts (so they are unit-testable without an MCP client); `build_server()`
wraps them into a FastMCP server and `main()` serves it over stdio.
"""

import io
from typing import Any, Dict, Optional

from ._version import __version__

# Full statevectors grow as 2**n; past this width return summarized
# probabilities instead of flooding the client with amplitudes.
MAX_STATEVECTOR_QUBITS = 10
TOP_PROBS = 20


def _parse_circuit(qasm: str):
    from .circuit_funcs.qasm_parser import from_qasm
    return from_qasm(qasm)


def run_circuit(qasm: str, shots: Optional[int] = None,
                backend: str = "tensornet") -> Dict[str, Any]:
    """Run an OpenQASM 2.0 circuit and return the result.

    With `shots`, returns measurement counts. Without, returns the full
    statevector for small circuits, or the largest basis-state probabilities
    for wide ones."""
    if backend not in ("tensornet", "statevector"):
        raise ValueError("backend must be 'tensornet' or 'statevector'.")
    c = _parse_circuit(qasm)
    if shots is not None:
        if not 0 < shots <= 100_000:
            raise ValueError('shots must be between 1 and 100000.')
        counts = c.run(backend=backend, shots=shots)
        return {"n_qubits": c.n_qubits, "shots": shots,
                "counts": dict(counts),
                "note": "bitstring order: leftmost character is the highest "
                        "qubit index; qubit 0 is the rightmost character."}
    state = c.run(backend=backend)
    if c.n_qubits <= MAX_STATEVECTOR_QUBITS:
        return {"n_qubits": c.n_qubits,
                "statevector": [[float(a.real), float(a.imag)]
                                for a in state.detach().numpy()],
                "note": "statevector[i] = [re, im] of basis state i; qubit 0 "
                        "is the least-significant bit of i."}
    probs = (state.abs() ** 2).detach()
    top = sorted(enumerate(probs.tolist()), key=lambda kv: -kv[1])[:TOP_PROBS]
    return {"n_qubits": c.n_qubits,
            "top_probabilities": {
                format(i, f'0{c.n_qubits}b'): p for i, p in top if p > 1e-12},
            "note": f"circuit wider than {MAX_STATEVECTOR_QUBITS} qubits: "
                    f"showing up to {TOP_PROBS} largest probabilities."}


def circuit_stats(qasm: str) -> Dict[str, Any]:
    """Qubit count, depth and gate counts of an OpenQASM 2.0 circuit."""
    c = _parse_circuit(qasm)
    return {"n_qubits": c.n_qubits, "depth": c.depth(),
            "gate_counts": dict(c.count_ops())}


def expectation_value(qasm: str, hamiltonian: str) -> Dict[str, Any]:
    """<psi|H|psi> for the circuit's final state.

    `hamiltonian` is a Pauli expression like "1.5*Z[0]*Z[1] - 0.5*X[0] + 2"
    (indices as [n] or directly after the letter; * is optional)."""
    from .utils import parse_hamiltonian
    c = _parse_circuit(qasm)
    h = parse_hamiltonian(hamiltonian)
    value = c.expect(h)
    return {"expectation_value": float(value),
            "hamiltonian": repr(h)}


def draw_circuit_png(qasm: str) -> bytes:
    """Render the circuit diagram and return PNG bytes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = _parse_circuit(qasm)
    c.run(backend='draw')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close('all')
    return buf.getvalue()


def eo_transpile(qasm: str) -> Dict[str, Any]:
    """Transpile a logical circuit to exchange-only spin-qubit pulses
    (3 physical spins per logical qubit) and summarize the pulse schedule."""
    import blueqat.eo  # registers the 'eo' backend
    from blueqat.eo import schedule_stats, to_schedule
    c = _parse_circuit(qasm)
    phys = c.run(backend='eo')
    sched = to_schedule(phys)
    stats = schedule_stats(sched)
    return {"logical_qubits": c.n_qubits,
            "physical_spins": phys.n_qubits,
            "n_pulses": int(stats["n_pulses"]),
            "serial_duration": stats["serial_duration"],
            "scheduled_duration": stats["scheduled_duration"],
            "parallel_speedup": stats["parallel_speedup"],
            "pulses_preview": sched["pulses"][:10]}


def cloud_run_circuit(qasm: str, shots: Optional[int] = None,
                      hamiltonian: Optional[str] = None,
                      mode: str = "tensornet") -> Dict[str, Any]:
    """Run a circuit on the Blueqat cloud (qapi.blueqat.app) instead of the
    local simulator. Needs a Blueqat API key (BLUEQAT_API_KEY or
    blueqat.cloud.save_api_key; get one at https://mcp.blueqat.app/login).

    With `shots`: measurement counts. With `hamiltonian` (Pauli expression):
    the expectation value. Otherwise: the statevector."""
    from . import cloud
    from .utils import parse_hamiltonian
    c = _parse_circuit(qasm)
    if hamiltonian is not None:
        value = c.run(backend='cloud', hamiltonian=parse_hamiltonian(hamiltonian),
                      mode=mode)
        return {"expectation_value": float(value)}
    if shots is not None:
        counts = c.run(backend='cloud', shots=shots, mode=mode)
        return {"counts": dict(counts), "shots": shots,
                "note": "bitstring order: qubit 0 is the rightmost character."}
    state = c.run(backend='cloud', mode=mode)
    return {"n_qubits": c.n_qubits,
            "statevector": [[float(z.real), float(z.imag)] for z in state.tolist()]}


def cloud_hardware_status() -> Dict[str, Any]:
    """Near-real-time status of the real quantum hardware behind the Blueqat
    cloud (public; no API key needed)."""
    from . import cloud
    return cloud.hardware_status()


def blueqat_info() -> Dict[str, Any]:
    """Version and capability summary of this blueqat installation."""
    return {
        "version": __version__,
        "simulation_modes": ["tensornet (default)", "statevector"],
        "circuit_format": "OpenQASM 2.0 (qelib1.inc gate set)",
        "hamiltonian_format": "Pauli expression, e.g. '1.5*Z[0]*Z[1] - 0.5*X[0]'",
        "extras": ["differentiable (PyTorch autograd)",
                   "exchange-only spin-qubit transpiler (eo_transpile)",
                   "circuit drawing (draw_circuit)"],
    }


def build_server():
    """Create the MCP server (requires the optional `mcp` dependency).

    Supports both the mcp 2.x high-level API (MCPServer) and the 1.x one
    (FastMCP) -- they share the tool()/run() surface used here."""
    try:
        from mcp.server.mcpserver import Image, MCPServer as _Server
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as _Server, Image
        except ImportError as e:
            raise ImportError(
                "The MCP server needs the optional dependency: "
                "pip install blueqat[mcp]") from e

    server = _Server(
        "blueqat",
        instructions=(
            "Quantum computing with the blueqat SDK. Circuits are OpenQASM "
            "2.0 text; qubit 0 is the least-significant bit of basis-state "
            "indices/bitstrings. Use run_circuit for states and sampling, "
            "expectation_value for Hamiltonians, draw_circuit for diagrams, "
            "and eo_transpile for exchange-only spin-qubit pulse compilation."))

    server.tool()(run_circuit)
    server.tool()(circuit_stats)
    server.tool()(expectation_value)
    server.tool()(eo_transpile)
    server.tool()(cloud_run_circuit)
    server.tool()(cloud_hardware_status)
    server.tool()(blueqat_info)

    @server.tool()
    def draw_circuit(qasm: str) -> Image:
        """Render an OpenQASM 2.0 circuit as a diagram image."""
        return Image(data=draw_circuit_png(qasm), format='png')

    return server


def main() -> None:
    """Entry point for the `blueqat-mcp` console script (stdio transport)."""
    build_server().run()


if __name__ == '__main__':
    main()
