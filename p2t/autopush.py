"""Periodic commit-and-push of the training run, without ever blocking it.

The user asked for the project to be pushed every five steps so progress survives
a closed laptop and a dead terminal.  Checkpoints, adapters and the raw credit
dumps are excluded by the repository's ``.gitignore``; only code, configs, tests
and the small ``report/`` artifacts travel.

Failure policy: pushing is best effort.  Every operation is wrapped, retried a
few times with backoff, and any remaining error is written to
``report/git_push.log`` and swallowed.  A flaky network must never cost a rollout.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

MAX_STAGED_BYTES = 50 * 1024 * 1024
PUSH_ATTEMPTS = 3


class AutoPusher:
    def __init__(self, *, enabled: bool, every: int, remote: str, branch: str,
                 report_dir: Path, repo_root: Path):
        self.enabled = bool(enabled and every > 0)
        self.every = max(1, int(every))
        self.remote = remote
        self.branch = branch
        self.report_dir = Path(report_dir)
        self.repo_root = Path(repo_root)
        self.log_path = self.report_dir / "git_push.log"
        self.lock_path = self.report_dir / ".git_push.lock"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def _run(self, *args, check=True, env=None):
        result = subprocess.run(["git", *args], cwd=self.repo_root, capture_output=True,
                                text=True, env={**os.environ, **(env or {})})
        if check and result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result

    def _log(self, entry: dict) -> None:
        try:
            with self.log_path.open("a") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
        except OSError:
            pass

    def maybe_push(self, step: int, metrics: dict) -> None:
        if not self.enabled or step % self.every:
            return
        self.push(step, metrics)

    # --------------------------------------------------------------------- push
    def push(self, step: int, metrics: dict) -> bool:
        """Commit and push once.  Returns success; never raises."""
        if not self.enabled:
            return False
        try:
            with self.lock_path.open("w") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    self._log({"step": step, "status": "skipped_locked", "at": time.time()})
                    return False
                return self._push_locked(step, metrics)
        except Exception as error:  # noqa: BLE001 - best effort by design
            self._log({"step": step, "status": "failed", "error": f"{type(error).__name__}: {error}",
                       "at": time.time()})
            return False

    def _push_locked(self, step: int, metrics: dict) -> bool:
        current = self._run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        if current != self.branch:
            self._log({"step": step, "status": "refused_wrong_branch", "branch": current,
                       "at": time.time()})
            return False
        self._run("add", "-A")
        staged = [name for name in self._run("diff", "--cached", "--name-only").stdout.split("\n")
                  if name]
        oversized = []
        for name in staged:
            path = self.repo_root / name
            try:
                if path.is_file() and path.stat().st_size > MAX_STAGED_BYTES:
                    oversized.append(name)
            except OSError:
                continue
        if oversized:
            self._run("reset")
            self._log({"step": step, "status": "refused_oversized", "files": oversized[:10],
                       "at": time.time()})
            return False
        if not staged:
            self._log({"step": step, "status": "nothing_to_commit", "at": time.time()})
            return False
        reward = metrics.get("raw_reward_mean")
        summary = f"raw reward {reward:.3f}" if isinstance(reward, (int, float)) else "n/a"
        message = (f"p2t step {step}: {summary}\n\n"
                   f"loss {metrics.get('loss')}, grad_norm {metrics.get('grad_norm')}, "
                   f"mean response tokens {metrics.get('mean_response_tokens')}\n\n"
                   "Co-Authored-By: Claude Code <noreply@anthropic.com>")
        self._run("commit", "-m", message, "-q")
        for attempt in range(PUSH_ATTEMPTS):
            result = self._run("push", self.remote, self.branch, check=False)
            if result.returncode == 0:
                sha = self._run("rev-parse", "HEAD").stdout.strip()
                self._log({"step": step, "status": "pushed", "sha": sha, "at": time.time()})
                return True
            time.sleep(2 ** attempt * 2)
        self._log({"step": step, "status": "push_failed",
                   "stderr": result.stderr.strip()[-500:], "at": time.time()})
        return False
