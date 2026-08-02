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
import stat
import urllib.error
import urllib.request
from collections import Counter as _Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .backends.backendbase import Backend, register_backend
from .gate import Operation

DEFAULT_ENDPOINT = "https://qapi.blueqat.app/v1"
ENV_API_KEY = "BLUEQAT_API_KEY"
REQUEST_TIMEOUT = 120.0

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
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    # API keys are secrets: restrict the file to its owner.
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return path


def delete_api_key() -> None:
    """Remove the stored API key from the config file (if present)."""
    path = config_path()
    data = _load_config_file()
    if "api_key" in data:
        del data["api_key"]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


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
        raise RuntimeError(f"Blueqat cloud error {e.code}: {detail or e.reason}") from None
    except urllib.error.URLError as e:
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
        coeff = float(coeff.real) if isinstance(coeff, complex) else float(coeff)
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
            # API counts are q0-first; convert back to SDK order.
            return _Counter({k[::-1]: v for k, v in result["counts"].items()})

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
                        confirm: bool = False) -> dict:
    """Submit a circuit to real quantum hardware.

    Requires `confirm=True`: hardware runs cost real money and are subject
    to your account's quota."""
    if not confirm:
        raise ValueError(
            "submit_hardware_job runs on real hardware and incurs real cost; "
            "pass confirm=True to proceed.")
    n, gate_list = circuit_to_gates(circuit)
    payload: Dict[str, Any] = {"n_qubits": n, "gates": gate_list,
                               "shots": int(shots), "confirm": True}
    if qpu_id is not None:
        payload["qpu_id"] = qpu_id
    return _request("POST", "/hardware/jobs", payload)


# Importing blueqat.cloud makes the backend available as backend='cloud'.
register_backend("cloud", CloudBackend, overwrite=True)
