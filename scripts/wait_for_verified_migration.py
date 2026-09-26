#!/usr/bin/env python3
"""Start the authorized ACP continuation only after its transfer is accepted."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path('/mnt/afs/202609/tyf-qwen-opd')
IMPORT = ROOT/'imports/20260927_ml2_base17_protocol'
RUN = ROOT/'runs/20260927v1_base17_grpo_migrated_n1_seed21_acp'
HOST = 'pt-5a50d1567d77437c94035024288a327f-worker-0'


def transfer_ready(folder):
    state_path = folder/'transfer_state_v2.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('status') == 'failed':
            raise RuntimeError(f'Transfer failed: {state.get("error")}')
    acceptance_path = folder/'transfer_acceptance.json'
    if not acceptance_path.exists():
        return False
    accepted = json.loads(acceptance_path.read_text())
    raw = (folder/'migration_manifest.json').read_bytes()
    files = json.loads(raw)['files']
    if (accepted.get('passed') is not True
            or accepted.get('manifest_sha256') != hashlib.sha256(raw).hexdigest()
            or accepted.get('files_verified') != len(files)
            or accepted.get('total_bytes') != sum(f['size_bytes'] for f in files)):
        raise ValueError('Incomplete or altered transfer acceptance')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout-hours', type=float, default=12)
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parents[1]
    commit = (runtime/'DEPLOYED_COMMIT').read_text().strip()
    if socket.gethostname() != HOST or runtime != ROOT/'deployments'/commit:
        raise ValueError('Only the frozen authorized ACP deployment may launch')
    if not 0 < args.timeout_hours <= 24:
        raise ValueError('Bounded transfer timeout required')
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'migration_wait.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN/'queue_manifest.json').exists():
            raise FileExistsError('Existing queue requires explicit recovery')
        state_path = RUN/'migration_wait_state.json'

        def state(value):
            temporary = state_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(pid=os.getpid(),source_commit=commit,**value),indent=2)+'\n')
            temporary.replace(state_path)

        deadline = time.monotonic()+3600*args.timeout_hours
        state(dict(status='waiting_for_verified_transfer',started_at_epoch=time.time()))
        try:
            while not transfer_ready(IMPORT):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Transfer deadline expired; no training or evaluation started')
                time.sleep(15)
            state(dict(status='launching_queue',verified_at_epoch=time.time()))
            entry = runtime/'scripts/run_acp_base17_migration.py'
            os.execv(sys.executable, [sys.executable, '-u', str(entry)])
        except BaseException as error:
            state(dict(status='failed',error=repr(error),failed_at_epoch=time.time()))
            raise


if __name__ == '__main__':
    main()
