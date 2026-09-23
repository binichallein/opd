#!/usr/bin/env python3
"""Detached inference-only ml2 queue; never starts or schedules training."""

from pathlib import Path
import socket

import qualify_qwen17_base_grpo as qualification
import run_qwen06_teacher_screen as historical


def configured_queue():
    run = qualification.private_module('_qwen17_base_grpo_screen_queue', historical.__file__)
    run.q = qualification.configured_qualifier()
    run.RUN = run.ROOT / 'runs/20260923v3_qwen17_base_grpo_teacher_screen_ml2'
    run.CACHE = Path('/dev/shm/q17gs0923')
    run.QUALIFIER_SCRIPT = 'qualify_qwen17_base_grpo.py'
    run.ASSET_DOWNLOADS = ()
    return run


if __name__ == '__main__':
    if socket.gethostname() != 'di-20260407234928-vrvxk':
        raise ValueError('This inference queue is authorized only on ml2')
    configured_queue().main()
