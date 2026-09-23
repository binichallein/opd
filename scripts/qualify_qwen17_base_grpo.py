#!/usr/bin/env python3
"""Inference-only public GRPO4B -> official Base1.7B teacher qualification.

Main views consume the exact completion-protocol student inputs and native EOS.
The teacher-native chat/stopping control is separate. No training is authorized.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import prepare_qwen17_pair_assets as models
import prepare_qwen06_screen_assets as historical_assets
import qualify_qwen06_teachers as historical

VERSION = 'qwen17_base_grpo_student_input_screen_v1'
NATIVE_STOP_IDS = [151645, 151643]


def private_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def specifications():
    student = models.MODELS['base_student']
    return {'b17': dict(repo=student['repo'], revision=student['revision'],
                        source=models._source_url(student), download_allowed=False,
                        files=student['files']),
            'g4': historical_assets.specifications()['g4']}


def configured_qualifier():
    # Separate namespaces retain the eight-pair screen's original configuration.
    q = private_module('_qwen17_base_grpo_qualification', historical.__file__)
    q.assets = private_module('_qwen17_base_grpo_assets', historical_assets.__file__)
    q.assets.specifications = specifications
    q.PAIRS = {'g4_b17': ('g4', 'b17')}
    q.STUDENTS = ('b17',)
    q.VERSION, q.STOP_IDS = VERSION, []
    original_protocol = q.protocol
    original_transfer = q.for_target
    original_invocation = q.invocation

    def protocol():
        result = original_protocol()
        result.update(main_prompt_protocol='qwen3_completion_boxed_v1',
                      main_eos_token_id=151643, native_control_stop_token_ids=NATIVE_STOP_IDS,
                      native_control_scope='Teacher native chat plus native-chat stops; not a template-only ablation')
        return result

    def for_target(requests, tokenizer, vocab_size):
        rows = original_transfer(requests, tokenizer, vocab_size)
        for row in rows:
            if row['input_source'] not in ('b17', 'g4'):
                raise ValueError('Unapproved input source')
            if row['input_source'] == 'b17' and tokenizer.eos_token_id != 151643:
                raise ValueError('Main comparison requires both models native EOS151643')
            row['sampling']['stop_token_ids'] = [] if row['input_source'] == 'b17' else list(NATIVE_STOP_IDS)
        return rows

    def invocation():
        result = original_invocation()
        for path in (Path(__file__), Path(__file__).with_name('run_qwen17_base_grpo_screen.py')):
            result['runtime_hashes'][str(path.resolve())] = q.old.sha256(path)
        result['runtime_sha256'] = q.old.object_hash(result['runtime_hashes'])
        result['script_sha256'] = q.old.sha256(__file__)
        return result

    q.protocol, q.for_target, q.invocation = protocol, for_target, invocation
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'generate', 'summarize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cell')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    q = configured_qualifier()
    if args.action == 'prepare':
        result = q.prepare(args.root)
    elif args.action == 'generate':
        result = q.generate(args.root, args.cell, args.gpu)
    else:
        result = q.summarize(args.root)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
