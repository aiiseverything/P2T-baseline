#!/usr/bin/env python3
"""User-authorized formal training, without a pilot or GPU preflight gate."""
from pathlib import Path
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import traceback

SUITE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    arm = sys.argv[1]
    if arm not in ('grpo', 'lam4'):
        raise ValueError('Only the two authorized Qwen arms are supported')
    plan = json.loads((SUITE / 'launch.json').read_text())
    source = Path(plan['source_snapshot'])
    sys.path.insert(0, str(source))
    from scripts import direct_rl_launcher as launcher
    from scripts.direct_rl_preflight import runtime_report
    from scripts.direct_rl_storage import verify_and_cleanup_exports
    family = Path(plan['frozen_family'])
    status = SUITE / 'qwen' / arm
    train = Path(plan['output_root']) / arm / 'train'
    with (status / 'arm.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if train.exists() or (status / 'command.json').exists():
                raise FileExistsError('Formal training already started; no implicit retry')
            if digest(family / 'experiment.json') != plan['frozen_experiment_sha256']:
                raise ValueError('Frozen experiment changed')
            manifest = launcher.read_manifest(family)
            identity = launcher.validation_identity(family)
            runtime = runtime_report(launcher.IMAGE)
            if (runtime['versions'] != launcher.RUNTIME or len(runtime['gpus']) != 3
                    or any(g['memory_bytes'] < 120 * 2**30 for g in runtime['gpus'])):
                raise ValueError('Runtime or GPU allocation differs from requested experiment')
            calibration_path = SUITE / 'shared-calibration.json'
            if digest(calibration_path) != plan['calibration_sha256']:
                raise ValueError('Shared calibration changed')
            calibration = launcher.canonical.validate_calibration(
                json.loads(calibration_path.read_text()), manifest['common_config'])
            Path(plan['output_root']).mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(plan['output_root']).free < 30 * 2**30:
                raise RuntimeError('Insufficient free space for final checkpoints')
            launcher.write_json(status / 'runtime.json', runtime)
            launcher.write_json(status / 'preflight-policy.json', {
                'status': 'skipped_by_explicit_user_request', 'gpu_preflight_passed': False,
                'calibration_source': plan['calibration_source'], 'sigma0': calibration['sigma0']})
            command = launcher.training_command(manifest, arm, train, sigma0=calibration['sigma0'])
            launcher.write_json(status / 'command.json', command)
            (status / 'stage').write_text('formal_training\n')
            # Actual training starts from a fresh LoRA; no pilot weights or RNG are reused.
            subprocess.run(command, cwd=status, check=True)
            completed = launcher.validate_completion(train, manifest, arm)
            if launcher.validation_identity(family) != identity:
                raise ValueError('Frozen source or model identity changed during training')
            cleanup = verify_and_cleanup_exports(train, 250)
            launcher.write_json(status / 'completion.json', {
                'status': 'complete', **completed, 'cleanup': cleanup,
                'frozen_experiment_sha256': plan['frozen_experiment_sha256']})
            (status / 'stage').write_text('complete\n')
        except BaseException as error:
            launcher.write_json(status / 'failure.json', {
                'status': 'failed', 'error': repr(error), 'traceback': traceback.format_exc()})
            (status / 'stage').write_text('failed\n')
            raise


if __name__ == '__main__':
    main()
