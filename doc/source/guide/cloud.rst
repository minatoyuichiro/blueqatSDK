Cloud access
============

:mod:`blueqat.cloud` submits circuits to the Blueqat cloud service at
``https://qapi.blueqat.app`` -- simulators on managed infrastructure and
real quantum hardware -- using the same ``Circuit`` API as local runs.

API keys
--------

Get a key at https://mcp.blueqat.app/login. Credentials resolve in this
order:

1. ``blueqat.cloud.configure(api_key=...)`` in the current process,
2. the ``BLUEQAT_API_KEY`` environment variable,
3. the config file ``~/.blueqat/config.json``.

.. code-block:: python

   import blueqat.cloud as cloud

   cloud.save_api_key("YOUR_API_KEY")   # persisted with owner-only (0600) permissions
   cloud.me()                           # account tier, limits and remaining quota

Running circuits on the cloud
-----------------------------

Importing :mod:`blueqat.cloud` registers the ``'cloud'`` backend. Results
follow the SDK's local conventions, so it is a drop-in replacement:

.. code-block:: python

   import blueqat.cloud
   from blueqat import Circuit
   from blueqat.utils import Z

   c = Circuit(2).h[0].cx[0, 1]
   c.m[:].run(backend='cloud', shots=100)          # Counter('00', '11', ...)
   c.run(backend='cloud')                          # statevector (torch.Tensor)
   c.run(backend='cloud', amplitude='11')          # a single amplitude
   c.run(backend='cloud', hamiltonian=1.0 * Z[0])  # expectation value

Named blocks and slices are expanded automatically for the wire format;
identity (constant) Hamiltonian terms are added back locally.

Other endpoints
---------------

.. code-block:: python

   cloud.health()                  # service health (no key needed)
   cloud.circuit_info(c)           # server-side validation / stats
   cloud.vqe_run(h, n_qubits=2)    # VQE on the cloud
   cloud.qaoa_run(qubo_terms)      # QAOA for a QUBO

Real quantum hardware
---------------------

.. code-block:: python

   cloud.hardware_status()         # near-real-time QPU status (public)
   cloud.hardware_qpus()           # available QPUs (authenticated)
   cloud.hardware_calibration()    # per-qubit error rates and coherence times
   cloud.hardware_next_window()    # when submissions are next accepted
   cloud.hardware_quote(shots=100, payer="me")   # cost, before committing

   job = cloud.submit_hardware_job(c, shots=100, confirm=True)

   cloud.hardware_jobs()                        # your recent submissions
   cloud.hardware_job(job["task_id"])           # status of one
   cloud.hardware_job_result(job["task_id"])    # counts, once finished
   cloud.cancel_hardware_job(job["task_id"])    # while still queued

``submit_hardware_job`` requires ``confirm=True``: hardware runs cost real
money and are subject to your account's quota. Pass ``preserve_layout=True``
to keep your qubit indices as written instead of letting the service remap
them.

When you don't know whether it ran
----------------------------------

A network error is not always a failure. The service sits behind Cloudflare,
whose 524 fires after about 100 seconds of *silence* from the origin -- that
bounds how long a reply may take to start, not how long the work may take. The
request has already arrived, and quite possibly finished. The same is true of a
client-side timeout.

Those cases raise :class:`~blueqat.cloud.CloudOutcomeUnknown` rather than a
plain error, so you can tell "this did not happen" from "I do not know whether
this happened":

.. code-block:: python

   try:
       job = cloud.submit_hardware_job(c, shots=100, confirm=True)
   except cloud.CloudOutcomeUnknown:
       # Do NOT resubmit blindly -- it may already be queued, and a duplicate
       # hardware job spends another slot and more money.
       for j in cloud.hardware_jobs()["jobs"]:
           print(j["task_id"], j["status"])

``CloudOutcomeUnknown`` subclasses ``RuntimeError``, so code that already
catches ``RuntimeError`` keeps working unchanged. A refused connection stays an
ordinary error: nothing was sent, so nothing can have run.

MCP integration
---------------

The bundled :doc:`MCP server <mcp>` exposes ``cloud_run_circuit`` and
``cloud_hardware_status`` tools, so an LLM client with your API key
configured can run circuits on the cloud, not just locally.

Testing
-------

The HTTP transport is injectable for hermetic tests::

   cloud.configure(transport=lambda method, path, payload, key, endpoint: {...})
