"""Connection retries are synthetic; tests never contact the paid endpoint."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import sys

import httpx
import pytest

SUITE = Path(__file__).resolve().parent


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location('resilient_judge_tested', SUITE / 'resilient_judge.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeClient:
    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        script = self.scripts.pop(0)
        return script(kwargs['extensions']['trace'])


def failure(error_type=httpx.ConnectError, phase='proxy.start_tls.failed', *, app=False, method=b'POST'):
    def invoke(trace):
        if phase is not None:
            trace('http11.send_request_headers.started', {'request': type('R', (), {'method': b'CONNECT'})()})
            if app:
                trace('http11.send_request_headers.started', {'request': type('R', (), {'method': method})()})
            trace(phase, {'exception': error_type('secret must never be logged')})
        raise error_type('secret must never be logged')
    return invoke


def success(trace):
    trace('http11.send_request_headers.started', {'request': type('R', (), {'method': b'POST'})()})
    trace('http11.send_request_headers.complete', {})
    return httpx.Response(200, request=httpx.Request('POST', 'https://example.invalid'), json={
        'choices': [{'message': {'content': '[[A=B]]'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}, 'id': 'response', 'model': 'gpt-4.1'})


def adapter(module, scripts):
    client, rows, sleeps = FakeClient(scripts), [], []
    return module.RetryingClient(client, rows.append, run_id='test-run', sleep=sleeps.append), client, rows, sleeps


def test_retry_only_failed_handshake_preserves_request_and_audits_four_attempts(module):
    wrapper, client, rows, sleeps = adapter(module, [failure(), failure(httpx.ConnectTimeout), failure(), success])
    request = {'model': 'gpt-4.1', 'messages': [{'role': 'user', 'content': 'unchanged'}]}
    assert wrapper.post(module.BASE_URL + '/chat/completions', json=request).status_code == 200
    assert len(client.calls) == 4 and sleeps == [1.0, 2.0, 4.0]
    assert all(call[1]['json'] is request for call in client.calls)
    assert [row['event'] for row in rows] == ['attempt_started', 'attempt_finished'] * 4
    finished = rows[1::2]
    assert [row['transport_attempt'] for row in finished] == [1, 2, 3, 4]
    assert [row['will_retry'] for row in finished] == [True, True, True, False]
    assert len({row['audit_id'] for row in rows}) == 1
    assert all(row['request_sha256'] == module.digest(request) for row in rows)
    assert all(row['run_id'] == 'test-run' and row['timestamp'] for row in rows)
    assert 'secret' not in json.dumps(rows)


@pytest.mark.parametrize('error_type', [httpx.ConnectError, httpx.ConnectTimeout])
@pytest.mark.parametrize('phase', ['connection.connect_tcp.failed', 'connection.start_tls.failed', 'proxy.start_tls.failed'])
def test_four_confirmed_handshake_failures_rethrow_original_class(module, error_type, phase):
    wrapper, client, rows, sleeps = adapter(module, [failure(error_type, phase)] * 5)
    with pytest.raises(error_type):
        wrapper.post(module.BASE_URL + '/chat/completions', json={})
    assert len(client.calls) == 4 and len(sleeps) == 3
    assert rows[-1]['retry_eligible'] is True and rows[-1]['will_retry'] is False
    assert rows[-1]['connection_failure_phase'] == phase


@pytest.mark.parametrize('error_type', [httpx.ReadTimeout, httpx.ReadError, httpx.WriteTimeout,
                                      httpx.WriteError, httpx.RemoteProtocolError, httpx.ProxyError,
                                      httpx.PoolTimeout, ValueError])
def test_other_errors_are_never_retried(module, error_type):
    wrapper, client, rows, sleeps = adapter(module, [failure(error_type)])
    with pytest.raises(error_type):
        wrapper.post(module.BASE_URL + '/chat/completions', json={})
    assert len(client.calls) == 1 and not sleeps
    assert rows[-1]['retry_eligible'] is False


@pytest.mark.parametrize('script', [failure(phase=None), failure(app=True),
                                    failure(phase='http11.receive_response_headers.failed')])
def test_unproven_connect_errors_become_non_retryable_record_type(module, script):
    wrapper, client, rows, sleeps = adapter(module, [script])
    with pytest.raises(module.UncertainTransportError, match='Connection failure was not proven'):
        wrapper.post(module.BASE_URL + '/chat/completions', json={})
    assert len(client.calls) == 1 and not sleeps
    assert rows[-1]['retry_eligible'] is False
    assert rows[-1]['raised_error_type'] == 'UncertainTransportError'


def test_unknown_application_method_is_fail_closed(module):
    trace = module.AttemptTrace()
    trace('http11.send_request_headers.started', {})
    trace('proxy.start_tls.failed', {})
    assert trace.application_headers_started is True
    assert not trace.eligible(httpx.ConnectError('ignored'))


def test_http_error_responses_never_retry(module):
    def rejected(trace):
        return httpx.Response(503, request=httpx.Request('POST', 'https://example.invalid'))
    wrapper, client, rows, sleeps = adapter(module, [rejected])
    response = wrapper.post(module.BASE_URL + '/chat/completions', json={})
    with pytest.raises(httpx.HTTPStatusError):
        response.raise_for_status()
    assert len(client.calls) == 1 and not sleeps
    assert rows[-1]['outcome'] == 'response' and rows[-1]['status_code'] == 503


def test_failure_to_persist_started_audit_prevents_dispatch(module):
    client = FakeClient([success])
    def broken(_):
        raise OSError('disk unavailable')
    wrapper = module.RetryingClient(client, broken, run_id='test-run')
    with pytest.raises(OSError):
        wrapper.post(module.BASE_URL + '/chat/completions', json={})
    assert not client.calls


def test_failure_to_persist_failed_attempt_prevents_retry(module):
    client = FakeClient([failure(), success])
    def broken(row):
        if row['event'] == 'attempt_finished':
            raise OSError('disk unavailable')
    wrapper = module.RetryingClient(client, broken, run_id='test-run')
    with pytest.raises(OSError):
        wrapper.post(module.BASE_URL + '/chat/completions', json={})
    assert len(client.calls) == 1


def test_threadsafe_audit_is_complete_jsonl(module, tmp_path):
    sink = module.AuditLog(tmp_path / 'attempts.jsonl')
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(lambda i: sink({'n': i}), range(256)))
    rows = [json.loads(line) for line in sink.path.read_text().splitlines()]
    assert sorted(row['n'] for row in rows) == list(range(256))


def test_make_relay_preserves_frozen_response_and_client_route(module, tmp_path, monkeypatch):
    (tmp_path / 'source').symlink_to(SUITE / 'source', target_is_directory=True)
    (tmp_path / 'resilient_policy.json').write_text(json.dumps({'transport_max_connect_attempts': 4}))
    captured = {}
    client = FakeClient([failure(), success])
    client.close = lambda: None
    def factory(**kwargs):
        captured.update(kwargs)
        return client
    monkeypatch.setattr(httpx, 'Client', factory)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    relay = module.make_relay('a-test-credential', suite=tmp_path)
    result = relay.judge_call({'model': 'gpt-4.1'})
    assert result['answer'] == '[[A=B]]' and result['usage']['total_tokens'] == 5
    assert captured['headers'] == {'Authorization': 'Bearer a-test-credential'}
    assert captured['trust_env'] is True
    limits = captured['limits']
    assert (limits.max_connections, limits.max_keepalive_connections, limits.keepalive_expiry) == (32, 32, 300)
    bindings = list((tmp_path / 'job/transport_bindings').glob('*.json'))
    assert len(bindings) == 1
    binding = json.loads(bindings[0].read_text())
    assert binding['transport_max_connect_attempts'] == 4
    assert binding['resilient_policy_sha256'] == module.sha256(tmp_path / 'resilient_policy.json')
    assert binding['frozen_judge_sha256'] == module.FROZEN_JUDGE_SHA256
    assert 'a-test-credential' not in ''.join(p.read_text() for p in (tmp_path / 'job').rglob('*.json*'))
    assert module.load_frozen(tmp_path).Relay is not type(relay)


def test_wrong_transport_policy_blocks_before_client_creation(module, tmp_path, monkeypatch):
    (tmp_path / 'resilient_policy.json').write_text(json.dumps({'transport_max_connect_attempts': 8}))
    monkeypatch.setattr(httpx, 'Client', lambda **kw: pytest.fail('must not construct client'))
    with pytest.raises(ValueError, match='transport_max_connect_attempts'):
        module.make_relay('test', suite=tmp_path)


def test_proxy_summary_records_loopback_port_without_userinfo(module, monkeypatch):
    monkeypatch.setenv('https_proxy', 'http://secret-user:secret-password@127.0.0.1:7890')
    monkeypatch.setenv('HTTPS_PROXY', 'http://secret-user:secret-password@private.example:17891')
    summary = module.proxy_summary()
    assert summary['https_proxy']['host'] == '127.0.0.1'
    assert summary['https_proxy']['port'] == 7890
    assert summary['HTTPS_PROXY']['host'] == '[non-loopback]'
    assert summary['HTTPS_PROXY']['port'] == 17891
    assert 'secret' not in json.dumps(summary) and 'private.example' not in json.dumps(summary)


def test_cli_forwards_original_arguments_and_restores_frozen_relay(module, monkeypatch):
    frozen = module.load_frozen(SUITE)
    original = frozen.Relay
    args = ['--questions', 'q', '--baseline-model', 'gpt-4o-mini-2024-07-18', '--dry-run', '--workers', '32']
    observed = {}
    def main():
        observed['args'] = sys.argv[1:]
        observed['replaced'] = frozen.Relay is not original
    monkeypatch.setattr(frozen, 'main', main)
    module.main(args)
    assert observed == {'args': args, 'replaced': True}
    assert frozen.Relay is original
