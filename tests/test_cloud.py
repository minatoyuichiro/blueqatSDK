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


# --- Gateway timeouts: "I don't know" is not "it failed" -------------------
#
# Cloudflare's 524 fires after ~100s of silence from the origin. That is a
# limit on how long a reply may take to *start*, not on how long the work may
# take: the request has already arrived and is very likely still running or
# already done. Reporting it as a plain failure invites a resubmission, and a
# resubmitted hardware job spends another scarce slot and more money.

def _raise_http(code, detail='timeout'):
    import io
    import urllib.error

    def transport(*args, **kwargs):
        raise urllib.error.HTTPError(
            'https://qapi.blueqat.app/v1/x', code, 'Gateway Timeout', {},
            io.BytesIO(json.dumps({'detail': detail}).encode()))
    return transport


@pytest.mark.parametrize('code', [504, 522, 523, 524])
def test_gateway_timeout_is_not_reported_as_failure(code, monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen', _raise_http(code))
    with pytest.raises(cloud.CloudOutcomeUnknown) as exc:
        cloud._http_transport('POST', '/hardware/jobs', {}, 'k', cloud.DEFAULT_ENDPOINT)
    msg = str(exc.value)
    assert str(code) in msg
    # The caller must be told not to blindly resubmit.
    assert 'not a failure' in msg and 'resubmit' in msg


def test_gateway_timeout_is_distinguishable_from_a_real_error(monkeypatch):
    """A caller must be able to tell the two apart with `except`."""
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen', _raise_http(500, 'boom'))
    with pytest.raises(RuntimeError) as exc:
        cloud._http_transport('POST', '/x', {}, 'k', cloud.DEFAULT_ENDPOINT)
    assert not isinstance(exc.value, cloud.CloudOutcomeUnknown)
    # ...while still being a RuntimeError, so existing handlers keep working.
    assert isinstance(exc.value, RuntimeError)
    assert issubclass(cloud.CloudOutcomeUnknown, RuntimeError)


def test_401_still_beats_the_timeout_branch(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen', _raise_http(401, ''))
    with pytest.raises(RuntimeError, match='rejected the API key'):
        cloud._http_transport('GET', '/me', None, 'k', cloud.DEFAULT_ENDPOINT)


@pytest.mark.parametrize('raised', ['bare', 'wrapped'])
def test_socket_timeout_is_also_an_unknown_outcome(raised, monkeypatch):
    """urlopen raises the socket timeout bare or wrapped in URLError depending
    on where it fires; both mean the request may already have landed."""
    import socket
    import urllib.error
    import urllib.request

    def transport(*args, **kwargs):
        if raised == 'bare':
            raise socket.timeout('timed out')
        raise urllib.error.URLError(socket.timeout('timed out'))

    monkeypatch.setattr(urllib.request, 'urlopen', transport)
    with pytest.raises(cloud.CloudOutcomeUnknown, match='resubmitting'):
        cloud._http_transport('POST', '/circuits/run', {}, 'k', cloud.DEFAULT_ENDPOINT)


def test_unreachable_host_is_still_a_plain_failure(monkeypatch):
    """A connection that never opened did *not* land -- keep it distinguishable."""
    import urllib.error
    import urllib.request

    def transport(*args, **kwargs):
        raise urllib.error.URLError(ConnectionRefusedError('refused'))

    monkeypatch.setattr(urllib.request, 'urlopen', transport)
    with pytest.raises(RuntimeError, match='Cannot reach') as exc:
        cloud._http_transport('GET', '/health', None, None, cloud.DEFAULT_ENDPOINT)
    assert not isinstance(exc.value, cloud.CloudOutcomeUnknown)


# --- Recovering from an unknown outcome ------------------------------------
#
# A gateway timeout is only actionable if the caller can find out what
# happened. These are the calls that answer that, so they are pinned to the
# paths and verbs the service actually publishes.

def test_hardware_jobs_lists_recent_submissions():
    t = FakeTransport({"jobs": [{"task_id": "abc", "status": "QUEUED"}]})
    cloud.configure(api_key="k", transport=t)
    assert cloud.hardware_jobs()["jobs"][0]["task_id"] == "abc"
    assert t.calls[0]["method"] == "GET"
    assert t.calls[0]["path"] == "/hardware/jobs?limit=20"
    cloud.hardware_jobs(limit=5)
    assert t.calls[1]["path"] == "/hardware/jobs?limit=5"


@pytest.mark.parametrize('call,args,method,path', [
    ('hardware_job', ('t1', ), 'GET', '/hardware/jobs/t1'),
    ('hardware_job_result', ('t1', ), 'GET', '/hardware/jobs/t1/result'),
    ('cancel_hardware_job', ('t1', ), 'POST', '/hardware/jobs/t1/cancel'),
    ('hardware_next_window', (), 'GET', '/hardware/qpus/next-window'),
    ('hardware_calibration', (), 'GET', '/hardware/qpus/calibration'),
])
def test_hardware_job_endpoints(call, args, method, path):
    t = FakeTransport({"status": "COMPLETED"})
    cloud.configure(api_key="k", transport=t)
    getattr(cloud, call)(*args)
    assert (t.calls[0]["method"], t.calls[0]["path"]) == (method, path)


def test_qpu_id_becomes_a_query_parameter():
    t = FakeTransport({})
    cloud.configure(api_key="k", transport=t)
    cloud.hardware_job("t1", qpu_id="lucy/sim")
    # Must be escaped: a raw '/' would change the path, not the query.
    assert t.calls[0]["path"] == "/hardware/jobs/t1?qpu_id=lucy%2Fsim"


def test_hardware_quote_asks_before_spending():
    t = FakeTransport({"cost": 12.5, "currency": "JPY"})
    cloud.configure(api_key="k", transport=t)
    assert cloud.hardware_quote(1000, payer="me")["cost"] == 12.5
    assert t.calls[0]["method"] == "POST" and t.calls[0]["path"] == "/hardware/quote"
    assert t.calls[0]["payload"] == {"shots": 1000, "payer": "me"}


def test_submit_sends_preserve_layout():
    t = FakeTransport({"task_id": "x"})
    cloud.configure(api_key="k", transport=t)
    cloud.submit_hardware_job(Circuit(1).h[0].m[:], shots=10, confirm=True)
    assert t.calls[0]["payload"]["preserve_layout"] is False
    cloud.submit_hardware_job(Circuit(1).h[0].m[:], shots=10, confirm=True,
                              preserve_layout=True)
    assert t.calls[1]["payload"]["preserve_layout"] is True


def test_all_job_calls_require_a_key():
    """None of the recovery calls may silently run unauthenticated: an
    unauthenticated list would look like "no jobs", i.e. "it did not land"."""
    cloud.reset_configuration()
    for call, args in [('hardware_jobs', ()), ('hardware_job', ('t', )),
                       ('hardware_job_result', ('t', )),
                       ('cancel_hardware_job', ('t', )),
                       ('hardware_quote', (10, 'me')),
                       ('hardware_next_window', ()),
                       ('hardware_calibration', ())]:
        with pytest.raises(RuntimeError, match='API key is not set'):
            getattr(cloud, call)(*args)


def test_timeout_message_names_the_recovery_call():
    """The warning is only useful if it says how to check."""
    import urllib.request
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(urllib.request, 'urlopen', _raise_http(524))
    try:
        with pytest.raises(cloud.CloudOutcomeUnknown) as exc:
            cloud._http_transport('POST', '/hardware/jobs', {}, 'k',
                                  cloud.DEFAULT_ENDPOINT)
        assert 'hardware_jobs()' in str(exc.value)
    finally:
        monkeypatch.undo()


def test_task_id_is_escaped_into_the_path():
    """An id is interpolated into the path; `quote` passes '/' through unless
    `safe=""`, so an unescaped one would address a different endpoint."""
    t = FakeTransport({})
    cloud.configure(api_key="k", transport=t)
    cloud.hardware_job("a/b")
    assert t.calls[0]["path"] == "/hardware/jobs/a%2Fb"


# --- the request headers themselves ----------------------------------------
#
# Every other test here goes through FakeTransport, which never builds a real
# request -- so nothing pinned the headers. The service sits behind Cloudflare,
# which rejects the "Python-urllib/x.y" signature with a 403 *before the
# request reaches the service at all*. Measured one variable at a time against
# the live endpoint: no header at all is 403, because urllib then supplies that
# signature itself; an empty agent is 200; a one-character agent is 200. So the
# rule is about that string, not about naming oneself -- but naming oneself is
# how you stay clear of it, and it is what the SDK does. Dropping the header in
# a tidy-up would break every call against the live endpoint while the whole
# suite stayed green, and the failure would read as "the service is down".

def _captured_request(method='GET', path='/health', payload=None, api_key=None):
    import urllib.request
    seen = {}

    class Response:
        status = 200
        def read(self): return b'{}'
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def urlopen(req, timeout=None):
        seen['request'] = req
        seen['timeout'] = timeout
        return Response()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    try:
        cloud._http_transport(method, path, payload, api_key, cloud.DEFAULT_ENDPOINT)
    finally:
        monkeypatch.undo()
    return seen


def test_requests_name_a_user_agent():
    seen = _captured_request()
    agent = seen['request'].get_header('User-agent')
    assert agent, "no User-Agent set: urllib would then send Python-urllib/x.y, which Cloudflare answers with 403"
    assert not agent.lower().startswith('python-urllib')
    assert 'blueqat' in agent.lower()


def test_user_agent_carries_the_version():
    from blueqat._version import __version__
    assert __version__ in _captured_request()['request'].get_header('User-agent')


def test_api_key_travels_as_a_bearer_token_only_when_present():
    with_key = _captured_request(api_key='secret')
    assert with_key['request'].get_header('Authorization') == 'Bearer secret'
    assert _captured_request(api_key=None)['request'].get_header('Authorization') is None


def test_a_payload_is_json_and_sets_its_content_type():
    seen = _captured_request('POST', '/circuits/run', {'shots': 4})
    assert json.loads(seen['request'].data.decode()) == {'shots': 4}
    assert seen['request'].get_header('Content-type') == 'application/json'
    assert seen['request'].get_method() == 'POST'
    # A GET carries no body and needs no content type.
    plain = _captured_request('GET', '/health')
    assert plain['request'].data is None
    assert plain['request'].get_header('Content-type') is None


def test_requests_carry_the_timeout():
    assert _captured_request()['timeout'] == cloud.REQUEST_TIMEOUT


# Both bodies below were captured from the live endpoint, not invented.
# Cloudflare content-negotiates its own errors, and the SDK always sends
# `Accept: application/json`, so it sees the JSON one -- a fabricated
# plain-text body made an earlier version of this test pass while the real
# service still produced an unhelpful message.
CLOUDFLARE_TEXT = b'error code: 1010\n'
CLOUDFLARE_JSON = json.dumps({
    "type": "https://developers.cloudflare.com/support/troubleshooting/"
            "http-status-codes/cloudflare-1xxx-errors/error-1010/",
    "title": "Error 1010: Access denied",
    "status": 403,
    "detail": "The site owner has blocked access based on your browser's signature.",
    "error_code": 1010,
    "error_name": "browser_signature_banned",
    "error_category": "access_denied",
    "cloudflare_error": True,
    "retryable": False,
}).encode()


def _raise_403(body):
    import io
    import urllib.error

    def transport(*args, **kwargs):
        raise urllib.error.HTTPError(
            'https://qapi.blueqat.app/v1/health', 403, 'Forbidden', {},
            io.BytesIO(body))
    return transport


@pytest.mark.parametrize('body', [CLOUDFLARE_TEXT, CLOUDFLARE_JSON])
def test_a_cloudflare_block_is_not_reported_as_a_permission_problem(body, monkeypatch):
    """A 403 from Cloudflare means the request never reached the service, so it
    says nothing about the key -- which is exactly what a bare 403 invites you
    to go and check."""
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen', _raise_403(body))
    with pytest.raises(RuntimeError) as exc:
        cloud._http_transport('GET', '/health', None, 'k', cloud.DEFAULT_ENDPOINT)
    message = str(exc.value)
    assert 'Cloudflare' in message
    assert 'not an authentication or permission problem' in message
    assert 'User-Agent' in message
    assert not isinstance(exc.value, cloud.CloudOutcomeUnknown)


def test_an_ordinary_403_still_reads_as_one():
    import io
    import urllib.error
    import urllib.request

    def transport(*args, **kwargs):
        raise urllib.error.HTTPError(
            'https://qapi.blueqat.app/v1/x', 403, 'Forbidden', {},
            io.BytesIO(json.dumps({'detail': 'quota exceeded'}).encode()))

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(urllib.request, 'urlopen', transport)
    try:
        with pytest.raises(RuntimeError, match='quota exceeded') as exc:
            cloud._http_transport('GET', '/x', None, 'k', cloud.DEFAULT_ENDPOINT)
    finally:
        monkeypatch.undo()
    assert 'Cloudflare' not in str(exc.value)
