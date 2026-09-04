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


class NotExamined(RuntimeError):
    """The request never got far enough to say anything about the circuit.

    A model reads what a tool returns, and a bare failure reads as a verdict on
    the input: shown ``HTTP 404`` or ``connection refused``, it concludes the
    circuit is at fault and starts rewriting a circuit that was never looked
    at. The message therefore says three things -- that the attempt failed,
    that the circuit itself is not implicated, and that it has not been checked
    -- and the third is the one that stops a rewrite.
    """


def _service_failure(action: str, error: BaseException) -> 'NotExamined':
    return NotExamined(
        f"Could not {action}: {error}. This is not a problem with the circuit, "
        f"and the circuit has NOT been checked -- it was never examined. Do not "
        f"change it in response to this. Retry, or run it locally with the "
        f"non-cloud tools, which need no service.")


def _through_the_service(action: str):
    """Wrap a call so a service failure is not read as a verdict on the input.

    A `ValueError` is left alone: the parser raising one really is about the
    circuit, and saying otherwise would be the mirror-image mistake.
    """
    def decorate(function):
        import functools

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except ValueError:
                raise                       # genuinely about the circuit
            except NotExamined:
                raise
            except Exception as error:
                from .cloud import CloudOutcomeUnknown
                if isinstance(error, CloudOutcomeUnknown):
                    # "It may already have run" is a different message from
                    # "it was never examined", and the difference matters: the
                    # advice below is to retry, which is the one thing not to
                    # do when the work may already be done and paid for.
                    raise
                raise _service_failure(action, error) from None

        return wrapper

    return decorate


@_through_the_service("reach the Blueqat cloud")
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


@_through_the_service("read the hardware status")
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
            "Quantum computing with the blueqat SDK, running locally. Every "
            "tool here takes the circuit as OpenQASM 2.0 text, which is what "
            "the 'qasm' in its name means; qubit 0 is the least-significant bit "
            "of basis-state indices/bitstrings. Use run_qasm for states and "
            "sampling, qasm_expectation for Hamiltonians, draw_qasm for "
            "diagrams, and qasm_to_eo_pulses for exchange-only spin-qubit pulse "
            "compilation."))

    # The tool names say that a circuit arrives as QASM text. blueqat's hosted
    # service exposes tools for the same jobs that take a JSON gate list
    # instead, and under bare names like `run_circuit` a client with both
    # registered cannot tell which it is calling -- the arguments differ, so it
    # finds out by being rejected. The Python functions keep their own names, so
    # importing this module is unaffected.
    server.tool(name="run_qasm")(run_circuit)
    server.tool(name="qasm_stats")(circuit_stats)
    server.tool(name="qasm_expectation")(expectation_value)
    server.tool(name="qasm_to_eo_pulses")(eo_transpile)
    server.tool(name="run_qasm_on_cloud")(cloud_run_circuit)
    server.tool(name="cloud_hardware_status")(cloud_hardware_status)
    server.tool(name="blueqat_info")(blueqat_info)

    @server.tool(name="draw_qasm")
    def draw_circuit(qasm: str) -> Image:
        """Render an OpenQASM 2.0 circuit as a diagram image."""
        return Image(data=draw_circuit_png(qasm), format='png')

    return server


def main() -> None:
    """Entry point for the `blueqat-mcp` console script (stdio transport)."""
    build_server().run()


if __name__ == '__main__':
    main()
