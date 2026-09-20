#!/usr/bin/env python3
"""Add three bounded read-only billing attempts around the frozen Arena runner.

Only RuntimeError from relay.usage is retried, with 1 and 2 second waits.
Paid judge calls, persisted records, monotonicity checks, and the cumulative
budget guard retain the original frozen runner's behavior.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


class BillingRetryRelay:
    def __init__(self, relay, *, sleep=time.sleep):
        self.relay = relay
        self.sleep = sleep

    def usage(self, start_date):
        for attempt in (1, 2, 3):
            try:
                return self.relay.usage(start_date)
            except RuntimeError:
                if attempt == 3:
                    raise RuntimeError('Read-only billing failed after 3 attempts; dispatch stopped') from None
                print(f'Read-only billing failed (attempt {attempt}/3); retrying in {attempt}s',
                      file=sys.stderr, flush=True)
                self.sleep(attempt)

    def judge_call(self, request):
        return self.relay.judge_call(request)

    def close(self):
        return self.relay.close()


def resume_suite(suite, *, policy, budget_cny, workers=32, max_requests=0,
                 relay_factory=None, sleep=time.sleep, continuation_dir=None):
    suite = Path(suite).resolve()
    selected = Path(continuation_dir or 'continuation-with-exclusions')
    continuation = (suite / selected).resolve()
    if continuation.parent != suite or not continuation.is_dir():
        raise ValueError('continuation directory must be an existing direct child of the suite')
    path = continuation / 'source/scripts/run_arena_with_exclusions.py'
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location('_frozen_arena_exclusion_resume', path)
        frozen = importlib.util.module_from_spec(spec)
        exec(compile(path.read_bytes(), str(path), 'exec'), frozen.__dict__)
        factory = relay_factory if relay_factory is not None else lambda: frozen.default_relay_factory(suite)
        return frozen.run_suite(suite, policy=policy, budget_cny=budget_cny,
            workers=workers, max_requests=max_requests,
            relay_factory=lambda: BillingRetryRelay(factory(), sleep=sleep))
    finally:
        sys.dont_write_bytecode = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--continuation-dir', type=Path)
    parser.add_argument('--budget-cny', type=float, required=True)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--max-requests', type=int, default=0)
    args = parser.parse_args(argv)
    result = resume_suite(args.suite, policy=args.policy, budget_cny=args.budget_cny,
                          workers=args.workers, max_requests=args.max_requests,
                          continuation_dir=args.continuation_dir)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Keep transport exceptions, headers and credentials out of diagnostics.
        print('Billing-retry judge stopped: ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
