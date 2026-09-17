#!/usr/bin/env python3
"""Additive, audited connection-establishment retries around the frozen judge.

Only ConnectError/ConnectTimeout with a traced TCP/TLS establishment failure
and no application request headers started can retry (four attempts total).
Proxy CONNECT headers are not application POST headers. Read/write/protocol
errors and HTTP responses never retry here. Unproven ConnectError is renamed
UncertainTransportError so record-level recovery cannot retry it accidentally.

The frozen prompt, request body, Relay response parsing, and cache identity
are unchanged. This module does not authorize record-level replacements.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid

import httpx

BASE_URL = 'https://api.linkapi.ai/v1'
FROZEN_JUDGE_SHA256 = 'aa0bbddb99d408e3d81156f63b8299381ed592a5f1abe2e252380ccb298d501f'
MAX_CONNECT_ATTEMPTS = 4
BACKOFF_SECONDS = (1.0, 2.0, 4.0)
CONNECTION_FAILURES = frozenset(('connection.connect_tcp.failed',
                                'connection.start_tls.failed', 'proxy.start_tls.failed'))
SCHEMA = 'arena_hard_connection_transport_v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exception_classes(error):
    """No exception messages, request objects, headers, addresses or credentials."""
    result, seen = [], set()
    while error is not None and id(error) not in seen and len(result) < 8:
        seen.add(id(error))
        name = type(error).__module__ + '.' + type(error).__name__
        result.append(name if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', name) else 'unknown.Exception')
        error = error.__cause__ or error.__context__
    return result


class UncertainTransportError(RuntimeError):
    """Cannot prove the application request was never sent; never auto-retry."""


class AttemptTrace:
    def __init__(self):
        self.events = []
        self.application_headers_started = False
        self.connection_failure_phase = None

    def __call__(self, name, info):
        safe_name = name if re.fullmatch(r'(connection|proxy|http11|http2)\.[a-z_]+\.(started|complete|failed)', name) else 'unrecognized'
        event = {'event': safe_name}
        if name.endswith('.send_request_headers.started'):
            method = getattr(info.get('request'), 'method', None)
            if isinstance(method, bytes):
                method = method.decode('ascii', errors='replace')
            event['method'] = method if method in ('CONNECT', 'POST', 'GET') else 'UNKNOWN'
            if method != 'CONNECT':
                self.application_headers_started = True
        if 'exception' in info:
            event['exception_classes'] = exception_classes(info['exception'])
        if name in CONNECTION_FAILURES:
            self.connection_failure_phase = name
        self.events.append(event)

    def eligible(self, error):
        return (isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout))
                and self.connection_failure_phase in CONNECTION_FAILURES
                and not self.application_headers_started)


class AuditLog:
    """Persist each event before progressing; lock across threads and processes."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def __call__(self, row):
        data = (json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode()
        with self.lock, self.path.open('ab', buffering=0) as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                stream.write(data)
                os.fsync(stream.fileno())
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class RetryingClient:
    def __init__(self, client, audit, *, run_id, sleep=None):
        self.client, self.audit, self.run_id = client, audit, run_id
        self.sleep = time.sleep if sleep is None else sleep

    def request(self, method, url, **kwargs):
        """Used by POST judging, also usable with a separate free-probe audit sink."""
        require(method in ('POST', 'GET'), 'Unsupported audited method')
        require(url == BASE_URL + ('/chat/completions' if method == 'POST' else '/models'),
                'Unexpected audited endpoint')
        require(not kwargs.get('extensions'), 'Caller trace extensions cannot override transport audit')
        request_sha = digest(kwargs.get('json') if method == 'POST' else {'method': method, 'url': url})
        audit_id = str(uuid.uuid4())
        for attempt in range(1, MAX_CONNECT_ATTEMPTS + 1):
            trace = AttemptTrace()
            common = {'schema': SCHEMA, 'run_id': self.run_id, 'audit_id': audit_id,
                      'request_sha256': request_sha, 'method': method, 'transport_attempt': attempt}
            self.audit({**common, 'event': 'attempt_started', 'timestamp': now()})
            started = time.monotonic()
            try:
                response = getattr(self.client, method.lower())(url, **{
                    **kwargs, 'extensions': {'trace': trace}})
            except Exception as error:
                eligible = trace.eligible(error)
                uncertain = isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)) and not eligible
                will_retry = eligible and attempt < MAX_CONNECT_ATTEMPTS
                self.audit({**common, 'event': 'attempt_finished', 'timestamp': now(),
                            'elapsed_seconds': round(time.monotonic() - started, 6),
                            'outcome': 'exception', 'exception_classes': exception_classes(error),
                            'raised_error_type': 'UncertainTransportError' if uncertain else type(error).__name__,
                            'events': trace.events, 'application_headers_started': trace.application_headers_started,
                            'connection_failure_phase': trace.connection_failure_phase,
                            'retry_eligible': eligible, 'will_retry': will_retry,
                            'backoff_seconds': BACKOFF_SECONDS[attempt - 1] if will_retry else 0})
                if uncertain:
                    raise UncertainTransportError('Connection failure was not proven to precede the application request') from None
                if not will_retry:
                    raise
                self.sleep(BACKOFF_SECONDS[attempt - 1])
                continue
            self.audit({**common, 'event': 'attempt_finished', 'timestamp': now(),
                        'elapsed_seconds': round(time.monotonic() - started, 6), 'outcome': 'response',
                        'status_code': response.status_code, 'events': trace.events,
                        'application_headers_started': trace.application_headers_started,
                        'connection_failure_phase': trace.connection_failure_phase,
                        'retry_eligible': False, 'will_retry': False, 'backoff_seconds': 0})
            return response
        raise AssertionError('Unreachable transport loop')

    def post(self, url, **kwargs):
        return self.request('POST', url, **kwargs)

    def get(self, url, **kwargs):
        # Read-only billing retains the frozen Relay's exact fail-closed behavior.
        return self.client.get(url, **kwargs)

    def close(self):
        return self.client.close()


@lru_cache(maxsize=8)
def load_frozen(suite):
    path = Path(suite).resolve() / 'source/scripts/judge_arena_hard.py'
    require(sha256(path) == FROZEN_JUDGE_SHA256, 'Frozen judge source hash mismatch')
    name = '_resilient_frozen_judge_' + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module._original_relay_class = module.Relay
    return module


def proxy_summary():
    result = {}
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        raw = os.environ.get(name)
        if raw:
            try:
                value = urlsplit(raw)
                loopback = value.hostname in ('localhost', '127.0.0.1', '::1')
                result[name] = {'present': True, 'scheme': value.scheme,
                                'loopback': loopback, 'host': value.hostname if loopback else '[non-loopback]',
                                'port': value.port,
                                'userinfo_present': bool(value.username or value.password)}
            except ValueError:
                result[name] = {'present': True, 'parseable': False}
    result['no_proxy_present'] = bool(os.environ.get('no_proxy') or os.environ.get('NO_PROXY'))
    return result


def make_relay(key, *, suite=Path(__file__).resolve().parent, timeout=600):
    """Factory API for the supervisor; accepts the unchanged frozen judge request."""
    require(isinstance(key, str) and bool(key.strip()), 'Missing relay credential')
    suite = Path(suite).resolve()
    policy_path = suite / 'resilient_policy.json'
    policy = json.loads(policy_path.read_text())
    require(policy.get('transport_max_connect_attempts') == MAX_CONNECT_ATTEMPTS,
            'resilient policy transport_max_connect_attempts must equal 4')
    frozen = load_frozen(suite)
    run_id = str(uuid.uuid4())
    import httpcore
    import httpcore._sync.connection
    binding = {'schema': SCHEMA, 'run_id': run_id, 'created_at': now(),
               'resilient_policy_sha256': sha256(policy_path),
               'transport_source_sha256': sha256(__file__), 'frozen_judge_sha256': FROZEN_JUDGE_SHA256,
               'transport_max_connect_attempts': MAX_CONNECT_ATTEMPTS, 'backoff_seconds': list(BACKOFF_SECONDS),
               'retry_exception_types': ['httpx.ConnectError', 'httpx.ConnectTimeout'],
               'eligible_failure_phases': sorted(CONNECTION_FAILURES),
               'require_no_application_headers_started': True, 'trust_env': True,
               'max_connections': 32, 'max_keepalive_connections': 32, 'keepalive_expiry_seconds': 300,
               'timeout_seconds': timeout, 'endpoint': BASE_URL, 'proxy_environment': proxy_summary(),
               'python': sys.version.split()[0], 'httpx': httpx.__version__, 'httpcore': httpcore.__version__,
               'httpx_transport_source_sha256': sha256(inspect.getfile(httpx.HTTPTransport.__init__)),
               'httpcore_connection_source_sha256': sha256(inspect.getfile(httpcore._sync.connection.HTTPConnection))}
    # The binding exists before a client (or any request) is created.
    frozen.atomic_json(suite / 'job/transport_bindings' / (run_id + '.json'), binding)
    client = httpx.Client(timeout=timeout, headers={'Authorization': f'Bearer {key}'}, trust_env=True,
                          limits=httpx.Limits(max_connections=32, max_keepalive_connections=32, keepalive_expiry=300))
    class ResilientRelay(frozen._original_relay_class):
        def __init__(self):
            self.client = RetryingClient(client, AuditLog(suite / 'job/connect_attempts.jsonl'), run_id=run_id)
            self.transport_binding = binding
    return ResilientRelay()


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--resilient-suite', type=Path, default=Path(__file__).resolve().parent)
    args, forwarded = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    frozen = load_frozen(args.resilient_suite.resolve())
    original_relay, original_argv = frozen.Relay, sys.argv
    try:
        frozen.Relay = lambda key, timeout=600: make_relay(key, suite=args.resilient_suite, timeout=timeout)
        sys.argv = [str(args.resilient_suite / 'source/scripts/judge_arena_hard.py'), *forwarded]
        frozen.main()
    finally:
        frozen.Relay, sys.argv = original_relay, original_argv


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # The source exception text could include transport or credential values.
        print('Resilient judge stopped: ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
