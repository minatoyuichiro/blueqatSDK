"""Tests for blueqat.cloud (the qapi.blueqat.app client).

Everything here is hermetic: HTTP is replaced by an injected transport.
Live-API tests run only when RUN_CLOUD_LIVE_TESTS=1 is set.
"""
import json
import math
import os
import stat

import pytest
import torch

import blueqat.cloud as cloud
from blueqat import Circuit
from blueqat.utils import X, Z


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point the config file at a temp dir and clear env/session state."""
    monkeypatch.setenv("BLUEQAT_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv(cloud.ENV_API_KEY, raising=False)
    cloud.reset_configuration()
    yield
    cloud.reset_configuration()


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, method, path, payload, api_key, endpoint):
        self.calls.append({"method": method, "path": path, "payload": payload,
                           "api_key": api_key, "endpoint": endpoint})
        return self.response


# --- credential resolution -------------------------------------------------------

def test_no_key_by_default():
    assert cloud.get_api_key() is None


def test_env_var_key(monkeypatch):
    monkeypatch.setenv(cloud.ENV_API_KEY, "env-key-123")
    assert cloud.get_api_key() == "env-key-123"


def test_save_and_load_key_file():
    path = cloud.save_api_key("file-key-456")
    assert cloud.get_api_key() == "file-key-456"
    assert json.loads(path.read_text())["api_key"] == "file-key-456"


def test_config_file_permissions_owner_only():
    path = cloud.save_api_key("secret-key")
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == (stat.S_IRUSR | stat.S_IWUSR), f"config file mode is {oct(mode)}"


def test_configure_beats_env_and_file(monkeypatch):
    cloud.save_api_key("file-key")
    monkeypatch.setenv(cloud.ENV_API_KEY, "env-key")
    cloud.configure(api_key="session-key")
    assert cloud.get_api_key() == "session-key"


def test_delete_api_key():
    cloud.save_api_key("to-be-deleted")
    cloud.delete_api_key()
    assert cloud.get_api_key() is None


def test_default_endpoint_is_qapi():
    assert cloud.get_endpoint() == "https://qapi.blueqat.app/v1"


def test_missing_key_raises_clear_error():
    with pytest.raises(RuntimeError, match="API key is not set"):
        Circuit(1).h[0].run(backend="cloud", shots=10)


# --- wire-format conversion --------------------------------------------------------

def test_circuit_to_gates():
    n, gates = cloud.circuit_to_gates(Circuit(2).h[0].rx(0.5)[1].cx[0, 1].m[:])
    assert n == 2
    assert gates == [
        {"gate": "h", "qubits": [0]},
        {"gate": "rx", "qubits": [1], "params": [0.5]},
        {"gate": "cx", "qubits": [0, 1]},
        {"gate": "measure", "qubits": [0]},
        {"gate": "measure", "qubits": [1]},
    ]


def test_circuit_to_gates_expands_blocks_and_slices():
    c = Circuit(3)
    with c.block("all-h"):
        c.h[:]
    n, gates = cloud.circuit_to_gates(c)
    assert n == 3
    assert [g["gate"] for g in gates] == ["h", "h", "h"]


def test_hamiltonian_to_terms_with_constant():
    terms, constant = cloud.hamiltonian_to_terms(1.5 * Z[0] * Z[1] - 0.5 * X[0] + 2)
    assert constant == pytest.approx(2.0)
    assert {"coeff": 1.5, "paulis": [{"op": "Z", "qubit": 0},
                                     {"op": "Z", "qubit": 1}]} in terms
    assert {"coeff": -0.5, "paulis": [{"op": "X", "qubit": 0}]} in terms


# --- CloudBackend request/response mapping ----------------------------------------

def test_cloud_counts_roundtrip_and_bit_order():
    # API returns q0-first bitstrings; the SDK must convert back to its own
    # q_{n-1}...q0 order for drop-in parity with local backends.
    t = FakeTransport({"counts": {"10": 60, "00": 40}, "shots": 100,
                       "bit_order": "bitstring[0] is qubit 0"})
    cloud.configure(api_key="k", transport=t)
    counts = Circuit(2).h[0].m[:].run(backend="cloud", shots=100)
    assert counts == {"01": 60, "00": 40}

    call = t.calls[0]
    assert call["method"] == "POST" and call["path"] == "/circuits/run"
    assert call["api_key"] == "k"
    assert call["payload"]["output"] == "counts"
    assert call["payload"]["shots"] == 100
    assert call["payload"]["n_qubits"] == 2
    assert call["payload"]["gates"][0] == {"gate": "h", "qubits": [0]}


def test_cloud_counts_honor_bit_order_argument():
    t = FakeTransport({"counts": {"10": 60, "00": 40}, "shots": 100,
                       "bit_order": "bitstring[0] is qubit 0"})
    cloud.configure(api_key="k", transport=t)
    # 'q0_first' asks for the API's own layout, so keys pass through unflipped.
    counts = Circuit(2).h[0].m[:].run(backend="cloud", shots=100, bit_order="q0_first")
    assert counts == {"10": 60, "00": 40}

    with pytest.raises(ValueError):
        Circuit(2).h[0].m[:].run(backend="cloud", shots=100, bit_order="little")


def test_cloud_counts_are_zero_padded_to_n_qubits():
    # A short key from the API ('1' meaning qubit 0) must not be reversed into a
    # differently-meaning short key; padding to n comes first.
    t = FakeTransport({"counts": {"1": 70, "0": 30}, "shots": 100,
                       "bit_order": "bitstring[0] is qubit 0"})
    cloud.configure(api_key="k", transport=t)
    counts = Circuit(4).x[0].m[:].run(backend="cloud", shots=100)
    assert counts == {"0001": 70, "0000": 30}


def test_cloud_statevector():
    t = FakeTransport({"statevector": [{"re": 1.0, "im": 0.0},
                                       {"re": 0.0, "im": 0.0}]})
    cloud.configure(api_key="k", transport=t)
    sv = Circuit(1).i[0].run(backend="cloud")
    assert torch.allclose(sv, torch.tensor([1, 0], dtype=torch.complex128))
    assert t.calls[0]["payload"]["output"] == "statevector"


def test_cloud_amplitude_reverses_bitstring():
    t = FakeTransport({"amplitude": {"re": 0.5, "im": -0.5}})
    cloud.configure(api_key="k", transport=t)
    amp = Circuit(2).h[0].run(backend="cloud", amplitude="10")
    assert amp == complex(0.5, -0.5)
    # SDK "10" (q1=1, q0=0) -> API q0-first "01"
    assert t.calls[0]["payload"]["amplitude"] == "01"


def test_cloud_expectation_adds_constant_locally():
    t = FakeTransport({"expectation": -1.25})
    cloud.configure(api_key="k", transport=t)
    value = Circuit(1).h[0].run(backend="cloud", hamiltonian=1.0 * Z[0] + 3.0)
    assert value == pytest.approx(-1.25 + 3.0)
    payload = t.calls[0]["payload"]
    assert payload["output"] == "expectation"
    assert payload["hamiltonian"] == [
        {"coeff": 1.0, "paulis": [{"op": "Z", "qubit": 0}]}]


def test_cloud_mode_passthrough():
    t = FakeTransport({"statevector": [{"re": 1.0, "im": 0.0},
                                       {"re": 0.0, "im": 0.0}]})
    cloud.configure(api_key="k", transport=t)
    Circuit(1).i[0].run(backend="cloud", mode="statevector")
    assert t.calls[0]["payload"]["mode"] == "statevector"


# --- REST helpers ------------------------------------------------------------------

def test_health_needs_no_key():
    t = FakeTransport({"status": "ok"})
    cloud.configure(transport=t)
    assert cloud.health() == {"status": "ok"}
    assert t.calls[0]["api_key"] is None


def test_me_requires_key():
    with pytest.raises(RuntimeError, match="API key is not set"):
        cloud.me()


def test_hardware_submit_requires_confirm():
    cloud.configure(api_key="k", transport=FakeTransport({}))
    with pytest.raises(ValueError, match="confirm=True"):
        cloud.submit_hardware_job(Circuit(1).h[0], shots=100)


def test_hardware_submit_payload():
    t = FakeTransport({"job_id": "j1"})
    cloud.configure(api_key="k", transport=t)
    out = cloud.submit_hardware_job(Circuit(1).h[0], shots=64, qpu_id="qpu:x",
                                    confirm=True)
    assert out == {"job_id": "j1"}
    payload = t.calls[0]["payload"]
    assert payload["confirm"] is True and payload["shots"] == 64
    assert payload["qpu_id"] == "qpu:x"
    assert t.calls[0]["path"] == "/hardware/jobs"


def test_repr_masks_key():
    cloud.configure(api_key="super-secret-api-key-value")
    r = repr(cloud.CloudBackend())
    assert "super-secret-api-key-value" not in r
    assert "supe" in r


# --- MCP tools over the cloud --------------------------------------------------------

def test_mcp_cloud_run_circuit_counts():
    from blueqat.mcp_server import cloud_run_circuit
    t = FakeTransport({"counts": {"01": 100}, "shots": 100, "bit_order": ""})
    cloud.configure(api_key="k", transport=t)
    out = cloud_run_circuit("h q[0]; cx q[0],q[1];", shots=100)
    assert out["counts"] == {"10": 100}


def test_mcp_cloud_run_circuit_expectation():
    from blueqat.mcp_server import cloud_run_circuit
    t = FakeTransport({"expectation": 0.921})
    cloud.configure(api_key="k", transport=t)
    out = cloud_run_circuit("rx(0.4) q[0];", hamiltonian="Z[0]")
    assert out["expectation_value"] == pytest.approx(0.921)


# --- opt-in live tests ----------------------------------------------------------------

live = pytest.mark.skipif(os.environ.get("RUN_CLOUD_LIVE_TESTS") != "1",
                          reason="set RUN_CLOUD_LIVE_TESTS=1 to hit the real API")


@live
def test_live_health():
    cloud.reset_configuration()
    assert cloud.health() == {"status": "ok"}


@live
def test_live_rejects_bad_key():
    cloud.reset_configuration()
    cloud.configure(api_key="definitely-not-a-real-key")
    with pytest.raises(RuntimeError, match="401"):
        Circuit(1).h[0].run(backend="cloud", shots=10)
