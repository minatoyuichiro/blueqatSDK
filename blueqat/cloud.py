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
"""API-key based access to the Blueqat cloud service (https://qapi.blueqat.app).

Credential resolution order:

1. An explicit `configure(api_key=...)` call in the current process.
2. The `BLUEQAT_API_KEY` environment variable.
3. The config file `~/.blueqat/config.json` (written by `save_api_key`,
   created with owner-only permissions).

Importing this module registers the `cloud` backend, so a circuit can be
submitted with the same API as local simulation::

    import blueqat.cloud
    Circuit(2).h[0].cx[0, 1].m[:].run(backend='cloud', shots=100)   # Counter
    Circuit(2).h[0].cx[0, 1].run(backend='cloud')                   # statevector
    Circuit(2).h[0].run(backend='cloud', hamiltonian=1.0 * Z[0])    # <psi|H|psi>

Results follow the SDK's conventions (Counter keys are q_{n-1}...q0 etc.),
so switching between local and cloud backends needs no code changes. Module
helpers cover the rest of the REST API: `health`, `me`, `circuit_info`,
`vqe_run`, `qaoa_run`, `hardware_status`, `hardware_qpus`,
`submit_hardware_job` (which requires `confirm=True` -- real hardware, real
cost).

Get an API key at https://mcp.blueqat.app/login.
"""

import json
import os
import socket
import stat
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections import Counter as _Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .backends.backendbase import Backend, apply_bit_order, register_backend
from .gate import Operation

DEFAULT_ENDPOINT = "https://qapi.blueqat.app/v1"
ENV_API_KEY = "BLUEQAT_API_KEY"
REQUEST_TIMEOUT = 120.0

#: Status codes meaning "the gateway gave up waiting", not "the work failed".
#: Cloudflare's 524 in particular fires after 100 seconds of silence from the
#: origin -- a limit on how long a reply may take to *start*, not on how long
#: the work may take. Treating one as a failure invites a resubmission, which
#: for a hardware job costs another slot and more money.
GATEWAY_TIMEOUTS = frozenset({504, 522, 523, 524})


class CloudOutcomeUnknown(RuntimeError):
    """The request's fate is unknown: it may have succeeded.

    Raised instead of a plain error when the connection or the gateway timed
    out, so that a caller can tell "this did not happen" apart from "I do not
    know whether this happened" -- and does not retry the second one blindly.
    """

_session: Dict[str, Any] = {"api_key": None, "endpoint": None, "transport": None}


# --- credentials ---------------------------------------------------------------

def config_path() -> Path:
    """Path of the persistent config file (override dir with BLUEQAT_CONFIG_DIR)."""
    base = os.environ.get("BLUEQAT_CONFIG_DIR")
    root = Path(base) if base else Path.home() / ".blueqat"
    return root / "config.json"


def _load_config_file() -> Dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_api_key(api_key: str, endpoint: Optional[str] = None) -> Path:
    """Persist the API key to the config file with owner-only permissions."""
    if not api_key or not isinstance(api_key, str):
        raise ValueError("api_key must be a non-empty string.")
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_config_file()
    data["api_key"] = api_key
    if endpoint is not None:
        data["endpoint"] = endpoint
    _write_private(path, data)
    return path


def _write_private(path: 'Path', data: Dict[str, Any]) -> None:
    """Write the config so that only its owner can ever read it.

    Creating the file with 0o600 rather than chmod-ing afterwards closes the
    window in which it exists at whatever the umask allows. On Windows
    ``os.chmod`` only touches the read-only bit and cannot make a file
    owner-only at all, so the key is written to a directory locked down instead,
    and a warning says so rather than leaving a false sense of protection.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    descriptor = os.open(path, flags, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    if os.name == 'nt':
        try:
            os.chmod(path.parent, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        except OSError:
            pass
        warnings.warn(
            f"{path} holds an API key, but Windows cannot restrict a file to its "
            f"owner through os.chmod. Check the file's ACL if this machine has "
            f"other users.", stacklevel=3)
    else:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def delete_api_key() -> None:
    """Remove the stored API key from the config file (if present)."""
    path = config_path()
    data = _load_config_file()
    if "api_key" in data:
        del data["api_key"]
        _write_private(path, data)


def get_api_key() -> Optional[str]:
    """Resolve the API key: configure() > environment > config file."""
    if _session["api_key"]:
        return _session["api_key"]
    env = os.environ.get(ENV_API_KEY)
    if env:
        return env
    return _load_config_file().get("api_key")


def get_endpoint() -> str:
    """Resolve the service endpoint: configure() > config file > default."""
    if _session["endpoint"]:
        return _session["endpoint"]
    return _load_config_file().get("endpoint", DEFAULT_ENDPOINT)


def configure(api_key: Optional[str] = None, endpoint: Optional[str] = None,
              transport: Optional[Callable[..., Any]] = None) -> None:
    """Set session-level cloud settings (highest priority, not persisted).

    `transport` is a callable ``(method, path, payload, api_key, endpoint)``
    returning the decoded JSON response; inject one for tests."""
    if api_key is not None:
        _session["api_key"] = api_key
    if endpoint is not None:
        _session["endpoint"] = endpoint
    if transport is not None:
        _session["transport"] = transport


def reset_configuration() -> None:
    """Clear session-level settings set by `configure` (env/file are untouched)."""
    _session["api_key"] = None
    _session["endpoint"] = None
    _session["transport"] = None


def _mask(key: str) -> str:
    return f"{key[:4]}...{key[-2:]}" if len(key) > 8 else "***"


# --- HTTP ------------------------------------------------------------------------

def _http_transport(method: str, path: str, payload: Optional[dict],
                    api_key: Optional[str], endpoint: str) -> Any:
    from ._version import __version__
    url = endpoint.rstrip('/') + path
    data = None
    # A real User-Agent matters: the service sits behind Cloudflare, which
    # blocks urllib's default "Python-urllib/x.y" signature.
    headers = {"Accept": "application/json",
               "User-Agent": f"blueqat-sdk/{__version__}"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode('utf-8')).get('detail', '')
        except Exception:
            detail = ''
        if e.code == 401:
            raise RuntimeError(
                "Blueqat cloud rejected the API key (401). "
                f"{detail or 'Get a key at https://mcp.blueqat.app/login.'}") from None
        if e.code in GATEWAY_TIMEOUTS:
            raise CloudOutcomeUnknown(
                f"The gateway stopped waiting for the Blueqat cloud after "
                f"{e.code} on {method} {path}. This is not a failure: the request "
                f"very likely arrived and may well have completed -- the gateway "
                f"gave up on the reply, not on the work. Do not resubmit before "
                f"checking whether it landed -- blueqat.cloud.hardware_jobs() lists "
                f"your recent submissions -- especially for a hardware job, "
                f"where resubmitting spends another slot and more money."
                + (f" {detail}" if detail else "")) from None
        raise RuntimeError(f"Blueqat cloud error {e.code}: {detail or e.reason}") from None
    except socket.timeout:
        raise CloudOutcomeUnknown(
            f"Timed out after {REQUEST_TIMEOUT:g}s waiting for {method} {path}. "
            f"The request may still have been received and acted on; check with "
            f"blueqat.cloud.hardware_jobs() before resubmitting.") from None
    except urllib.error.URLError as e:
        if isinstance(e.reason, socket.timeout):
            raise CloudOutcomeUnknown(
                f"Timed out after {REQUEST_TIMEOUT:g}s waiting for {method} {path}. "
                f"The request may still have been received and acted on; check with "
                f"blueqat.cloud.hardware_jobs() before resubmitting.") from None
        raise RuntimeError(f"Cannot reach the Blueqat cloud at {url}: {e.reason}") from None


def _request(method: str, path: str, payload: Optional[dict] = None,
             auth: bool = True) -> Any:
    api_key = get_api_key() if auth else None
    if auth and not api_key:
        raise RuntimeError(
            "Blueqat cloud API key is not set. Set the BLUEQAT_API_KEY "
            "environment variable, call blueqat.cloud.save_api_key(...), or "
            "blueqat.cloud.configure(api_key=...). "
            "Get a key at https://mcp.blueqat.app/login.")
    transport = _session["transport"] or _http_transport
    return transport(method, path, payload, api_key, get_endpoint())


# --- wire-format conversion --------------------------------------------------------

def circuit_to_gates(circuit) -> Tuple[int, List[dict]]:
    """Convert a Circuit into the API's gate-list format:
    ``[{"gate": "h", "qubits": [0]}, {"gate": "rx", "qubits": [0], "params": [0.5]}, ...]``
    (slices and named blocks are expanded)."""
    from .circuit_funcs.flatten import flatten
    from .gate import Measurement
    flat = flatten(circuit)
    gates = []
    for op in flat.ops:
        targets = op.targets
        qubits = [targets] if isinstance(targets, int) else list(targets)
        entry: Dict[str, Any] = {"gate": op.lowername, "qubits": qubits}
        if op.params:
            entry["params"] = [float(p) for p in op.params]
        if isinstance(op, Measurement) and op.key is not None:
            options: Dict[str, Any] = {"key": op.key}
            if op.duplicated is not None:
                options["duplicated"] = op.duplicated
            entry["options"] = options
        gates.append(entry)
    return flat.n_qubits, gates


def hamiltonian_to_terms(hamiltonian) -> Tuple[List[dict], float]:
    """Convert a Pauli Expr/Term into the API's term list
    ``[{"coeff": c, "paulis": [{"op": "X", "qubit": 0}, ...]}, ...]``.

    Identity (constant) terms can't be sent over the wire; they are returned
    separately as a float to add to the expectation value locally."""
    expr = hamiltonian.to_expr().simplify()
    terms = []
    constant = 0.0
    for term in expr:
        coeff = term.coeff
        if isinstance(coeff, complex):
            # A Hamiltonian is Hermitian, so a Pauli term with an imaginary
            # coefficient is not one. Dropping the imaginary part quietly sends a
            # different operator than the caller wrote.
            if abs(coeff.imag) > 1e-12:
                raise ValueError(
                    f"Term {term!r} has a complex coefficient ({coeff}); a "
                    f"Hamiltonian must be Hermitian. Check for a missing "
                    f"conjugate, or use .simplify() on the expression.")
            coeff = float(coeff.real)
        else:
            coeff = float(coeff)
        if not term.ops:
            constant += coeff
            continue
        terms.append({"coeff": coeff,
                      "paulis": [{"op": op.op, "qubit": op.n} for op in term.ops]})
    return terms, constant


def _from_jsonable_complex(z: Any) -> complex:
    if isinstance(z, dict):
        return complex(z.get("re", 0.0), z.get("im", 0.0))
    if isinstance(z, (list, tuple)) and len(z) == 2:
        return complex(z[0], z[1])
    return complex(z)


# --- the backend ----------------------------------------------------------------------

class CloudBackend(Backend):
    """Backend submitting circuits to https://qapi.blueqat.app (POST
    /v1/circuits/run). Results are converted back to SDK conventions, so it
    is a drop-in replacement for the local backends."""

    def run(self, gates: List[Operation], n_qubits: int, *args: Any, **kwargs: Any) -> Any:
        from .circuit import Circuit
        n, gate_list = circuit_to_gates(Circuit(n_qubits, list(gates)))
        n = max(n, n_qubits, 1)
        payload: Dict[str, Any] = {
            "n_qubits": n,
            "gates": gate_list,
            "mode": kwargs.get("mode", "tensornet"),
        }

        hamiltonian = kwargs.get("hamiltonian")
        amplitude = kwargs.get("amplitude")
        shots = kwargs.get("shots")

        if hamiltonian is not None:
            terms, constant = hamiltonian_to_terms(hamiltonian)
            payload["output"] = "expectation"
            payload["hamiltonian"] = terms
            result = _request("POST", "/circuits/run", payload)
            return float(result["expectation"]) + constant

        if amplitude is not None or kwargs.get("returns") == "amplitude":
            target = amplitude if amplitude is not None else "0" * n
            payload["output"] = "amplitude"
            # SDK bitstrings are q_{n-1}...q0; the public API is q0-first.
            payload["amplitude"] = str(target)[::-1]
            result = _request("POST", "/circuits/run", payload)
            return _from_jsonable_complex(result["amplitude"])

        if shots is not None:
            payload["output"] = "counts"
            payload["shots"] = int(shots)
            result = _request("POST", "/circuits/run", payload)
            # API counts are q0-first, so reversing puts them in SDK order; only then
            # can they be zero-padded (padding a q0-first key on the left would move
            # its qubits). `apply_bit_order` does the padding and, if the caller asked
            # for `bit_order="q0_first"`, flips the padded key back.
            bit_order = kwargs.get("bit_order", "q0_last")
            sdk_order = _Counter({k[::-1]: v for k, v in result["counts"].items()})
            return apply_bit_order(sdk_order, n, bit_order)

        payload["output"] = "statevector"
        result = _request("POST", "/circuits/run", payload)
        import torch
        return torch.tensor([_from_jsonable_complex(z) for z in result["statevector"]],
                            dtype=torch.complex128)

    def __repr__(self) -> str:
        key = get_api_key()
        status = f"api_key={_mask(key)}" if key else "unconfigured"
        return f"CloudBackend({status}, endpoint={get_endpoint()!r})"


# --- REST convenience helpers ------------------------------------------------------------

def health() -> dict:
    """Service health (no authentication required)."""
    return _request("GET", "/health", auth=False)


def me() -> dict:
    """The authenticated account: tier, limits and remaining quota."""
    return _request("GET", "/me")


def circuit_info(circuit) -> dict:
    """Server-side circuit validation and stats without running it."""
    n, gate_list = circuit_to_gates(circuit)
    return _request("POST", "/circuits/info", {"n_qubits": n, "gates": gate_list})


def vqe_run(hamiltonian, n_qubits: int, layers: int = 1) -> dict:
    """Run VQE on the cloud for a Pauli Hamiltonian."""
    terms, constant = hamiltonian_to_terms(hamiltonian)
    result = _request("POST", "/vqe/run",
                      {"n_qubits": n_qubits, "hamiltonian": terms, "layers": layers})
    if constant and isinstance(result, dict) and "energy" in result:
        result = dict(result)
        result["energy"] = result["energy"] + constant
    return result


def qaoa_run(qubo: List[dict], steps: int = 1, shots: int = 256) -> dict:
    """Run QAOA on the cloud for a QUBO given as
    ``[{"i": 0, "j": 1, "value": 2.0}, ...]`` (see the API docs)."""
    return _request("POST", "/qaoa/run",
                    {"qubo": qubo, "steps": steps, "shots": shots})


def hardware_status() -> dict:
    """Near-real-time hardware status snapshot."""
    return _request("GET", "/hardware/status", auth=False)


def hardware_qpus() -> dict:
    """List available QPUs (authenticated)."""
    return _request("GET", "/hardware/qpus")


def submit_hardware_job(circuit, shots: int, qpu_id: Optional[str] = None,
                        confirm: bool = False,
                        preserve_layout: bool = False) -> dict:
    """Submit a circuit to real quantum hardware.

    Requires `confirm=True`: hardware runs cost real money and are subject
    to your account's quota."""
    if not confirm:
        raise ValueError(
            "submit_hardware_job runs on real hardware and incurs real cost; "
            "pass confirm=True to proceed.")
    n, gate_list = circuit_to_gates(circuit)
    payload: Dict[str, Any] = {"n_qubits": n, "gates": gate_list,
                               "shots": int(shots), "confirm": True,
                               "preserve_layout": bool(preserve_layout)}
    if qpu_id is not None:
        payload["qpu_id"] = qpu_id
    return _request("POST", "/hardware/jobs", payload)


def _quote(v) -> str:
    """Escape a path segment. `safe=""` matters: `quote` passes '/' through by
    default, so an id containing one would silently change the path."""
    return urllib.parse.quote(str(v), safe="")


def _qpu_query(qpu_id: Optional[str]) -> str:
    return f"?qpu_id={_quote(qpu_id)}" if qpu_id is not None else ""


def hardware_jobs(limit: int = 20) -> dict:
    """List your recent hardware jobs, newest first.

    This is the way to answer "did my submission actually land?" after a
    `CloudOutcomeUnknown` -- check here before resubmitting, since a duplicate
    hardware job spends another slot and more money."""
    return _request("GET", f"/hardware/jobs?limit={int(limit)}")


def hardware_job(task_id: str, qpu_id: Optional[str] = None) -> dict:
    """Status of one hardware job."""
    return _request("GET", f"/hardware/jobs/{_quote(task_id)}{_qpu_query(qpu_id)}")


def hardware_job_result(task_id: str, qpu_id: Optional[str] = None) -> dict:
    """Result of a finished hardware job."""
    return _request("GET", f"/hardware/jobs/{_quote(task_id)}/result{_qpu_query(qpu_id)}")


def cancel_hardware_job(task_id: str, qpu_id: Optional[str] = None) -> dict:
    """Cancel a queued hardware job."""
    return _request("POST", f"/hardware/jobs/{_quote(task_id)}/cancel{_qpu_query(qpu_id)}")


def hardware_quote(shots: int, payer: str) -> dict:
    """What a hardware run would cost, before committing to it."""
    return _request("POST", "/hardware/quote",
                    {"shots": int(shots), "payer": str(payer)})


def hardware_next_window(qpu_id: Optional[str] = None) -> dict:
    """When the next hardware submission window opens.

    Hardware is not always accepting jobs; a submission outside a window
    queues until the next one."""
    return _request("GET", f"/hardware/qpus/next-window{_qpu_query(qpu_id)}")


def hardware_calibration(qpu_id: Optional[str] = None) -> dict:
    """Current per-qubit calibration data (error rates, coherence times)."""
    return _request("GET", f"/hardware/qpus/calibration{_qpu_query(qpu_id)}")


# Importing blueqat.cloud makes the backend available as backend='cloud'.
register_backend("cloud", CloudBackend, overwrite=True)
