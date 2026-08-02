MCP server (LLM integration)
============================

blueqat ships an `MCP (Model Context Protocol)
<https://modelcontextprotocol.io/>`_ server, so LLM clients like Claude
Desktop and Claude Code can build, run, analyze and draw quantum circuits
through natural language.

Setup
-----

.. code-block:: console

   pip install blueqat[mcp]

Then register the ``blueqat-mcp`` command with your MCP client. For Claude
Desktop, add to the config file:

.. code-block:: json

   { "mcpServers": { "blueqat": { "command": "blueqat-mcp" } } }

For Claude Code:

.. code-block:: console

   claude mcp add blueqat -- blueqat-mcp

Tools
-----

``run_circuit(qasm, shots=None, backend="tensornet")``
   Run an OpenQASM 2.0 circuit. Returns the statevector for small circuits,
   the largest basis-state probabilities for wide ones, or measurement
   counts when ``shots`` is given.

``circuit_stats(qasm)``
   Qubit count, depth and gate counts.

``expectation_value(qasm, hamiltonian)``
   :math:`\langle\psi|H|\psi\rangle` with the Hamiltonian written as a Pauli
   expression such as ``"1.5*Z[0]*Z[1] - 0.5*X[0] + 2"``.

``draw_circuit(qasm)``
   The circuit diagram as a PNG image.

``eo_transpile(qasm)``
   Compile the logical circuit to exchange-only spin-qubit pulses
   (see :doc:`exchange_only`) and summarize the pulse schedule.

``blueqat_info()``
   Version and capability summary.

Safety
------

Tool inputs are never executed as code: circuits go through blueqat's
eval-free OpenQASM parser and Hamiltonians through
:func:`blueqat.utils.parse_hamiltonian`, a small regex-based parser.
Statevector responses are capped (wide circuits return summarized
probabilities) so a single call can't flood the client.
