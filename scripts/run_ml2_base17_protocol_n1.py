#!/usr/bin/env python3
"""ml2: a completion-protocol repeat of the completed historical 100-step pair."""

from functools import partial
import math
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import run_acp_qwen17_n1 as original
import run_qwen17_base_grpo_pair as base_pair
from opd_ext.math_protocol import QWEN_COMPLETION_PROTOCOL, completion_input_ids, generated_think_tags

ROOT = base_pair.ROOT
BASELINE_ROOT = ROOT / 'runs/20260925v4_historical17_pair_n1_step100_save25_seed21_ml2'
BASELINE_COMMIT = '7bf5420d69d5e3ce23cb79882a0ceb6c391b1cf5'
RUN_ROOT = ROOT / 'runs/20260926v1_qwen17_base_grpo_protocol_n1_seed21_ml2'
STUDENT = Path('/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/models/Qwen3-1.7B-Base')
TEACHER = ROOT / 'models/Qwen3-4B-Base-GRPO'
PROJECT = 'opd_ml2_base17_protocol_n1'


def validate_location(host, root, fstype):
    if host != 'di-20260407234928-vrvxk' or root != ROOT or fstype != 'nfs4':
        raise ValueError('Only the verified ml2 worker and persistent NFS root are authorized')


def configure(run):
    q = run.q
    q.ROOT, q.RUN_ROOT, q.CACHE = ROOT, RUN_ROOT, run.CACHE
    q.STUDENT, q.TEACHER = STUDENT, TEACHER
    q.STUDENT_REVISION = q.TEACHER_REVISION = ''
    q.STEPS, q.SAVE_STEPS = run.STEPS, run.SAVE_STEPS
    q.PROJECT_NAME, q.EXPERIMENT_PREFIX = PROJECT, 'ml2-base17-protocol-n1'
    q.PROTOCOL = q.EVALUATION_PROTOCOL = QWEN_COMPLETION_PROTOCOL
    q.LOSS_REFERENCE = ROOT / 'deployments' / BASELINE_COMMIT
    q.BASELINE_ALIGNMENT = 'Protocol-only repeat of matched historical 100-step32x1; legacy losses and request seed21 unchanged'
    q.base.PLOT_PYTHON = q.base.PYTHON


def training_env(run, runtime, commit, root, variant, probe=None):
    env = run.q.training_env(runtime, commit, root, variant, probe)
    env.update(REMOTE='ml2', TRAIN_BATCH_SIZE='32', ROLLOUT_GROUP_SIZE='1',
        TOTAL_TRAINING_STEPS='100', STOP_AFTER_STEP=str(probe or -1),
        OPD_DIAG_INTERVAL='1' if probe else '5', OPD_REQUEST_SEED_RULE='legacy', RAY_NUM_CPUS='64')
    return env


def expected_card(run, runtime, commit, root, variant, probe=None):
    card = run.q.expected_card(root, variant, commit, probe)
    card.update(train_batch_size=32, rollout_group_size=1, total_training_steps=100,
                stop_after_step=probe or -1, request_seed_rule='legacy', ray_num_cpus=64,
                opd_diag_interval=1 if probe else 5, opd_block_ablation='legacy')
    return card


def validate_baseline_card(expected, actual):
    allowed = {'source_commit','opd_prompt_protocol','project_name','experiment_name',
               'diagnostic_output_dir','lossless_rollout_dir','baseline_alignment'}
    original.require_equal_card({k:v for k,v in expected.items() if k not in allowed}, actual)


def validate_rollout(row, prompt_ids, step):
    expected = dict(temperature=1.,top_p=.9,top_k=-1,max_tokens=16384,n=1,
                    ignore_eos=False,stop_token_ids=[],seed=21)
    if (row['protocol'] != QWEN_COMPLETION_PROTOCOL or row['enable_thinking'] is not False
            or row['step'] != step or row['mask_policy'] != 'historical_eos_mask'
            or row['eos_token_id'] != 151643 or row.get('request_identity') is not None
            or any(row['sampling'].get(k) != v for k,v in expected.items())):
        raise ValueError('Completion protocol must retain the historical request seed rule')
    ids, count = row['training_response_token_ids'], row['response_length']
    first_eos = ids.index(151643)+1 if 151643 in ids else len(ids)
    mask = [1]*first_eos + [0]*(len(ids)-first_eos)
    if (row['prompt_token_ids'] != prompt_ids or len(ids) != 16384
            or row['response_tensor_width'] != 16384 or not 0 < count <= 16384
            or count != len(row['response_token_ids']) or ids[:count] != row['response_token_ids']
            or row['padding_length'] != 16384-count or mask != row['training_response_mask']
            or row['response_mask'] != mask[:count] or mask != [1]*count+[0]*(16384-count)
            or len(row['training_rollout_log_probs']) != 16384
            or row['rollout_log_probs'] != row['training_rollout_log_probs'][:count]
            or not all(math.isfinite(v) for v in row['training_rollout_log_probs'])):
        raise ValueError('Completion input, actual length, EOS mask or logprob differs')
    if row['finish_reason'] == 'length':
        valid_stop = count == 16384 and row['stop_reason'] is None
    else:
        valid_stop = (row['finish_reason'] == 'stop' and row['response_token_ids'][-1] == 151643
                      and row['stop_reason'] is None)
    if not valid_stop:
        raise ValueError('Unexpected completion termination')


def preflight(run, runtime):
    q = run.q
    protected = q.common_preflight(runtime, run.LOSS_OVERRIDES)
    state = q.base.read_json(BASELINE_ROOT/'queue_state.json')
    accepted = q.base.read_json(BASELINE_ROOT/'evaluation_acceptance.json')
    if state.get('status') != 'complete' or not accepted.get('complete') or len(accepted.get('models',{})) != 8:
        raise ValueError('All eight historical comparison evaluations are required')
    paths = [BASELINE_ROOT/name for name in ('queue_state.json','evaluation_acceptance.json',
        'per_benchmark_results.json','paired_rollout_acceptance.json','input_plan.json','environment.json')]
    environment = q.base.read_json(BASELINE_ROOT/'environment.json')
    if environment['python'] != sys.version or environment['packages'] != q.qualify.runtime_versions():
        raise ValueError('Historical training environment differs')
    for variant in run.VARIANTS:
        old = q.base.read_json(BASELINE_ROOT/variant/'run_card.json')
        if old['source_commit'] != BASELINE_COMMIT or old['opd_prompt_protocol'] != 'qwen3_historical17_v1':
            raise ValueError('Wrong historical control')
        validate_baseline_card(run.expected_card(runtime,runtime.name,RUN_ROOT,variant),old)
        manifest = BASELINE_ROOT/variant/'artifact_hashes.sha256'
        protected.update({str(p):h for h,p in q.base.paired.load_sha256_manifest(manifest)})
        paths += [manifest,BASELINE_ROOT/variant/'run_card.json']
    if run.n1.input_plan(q) != q.base.read_json(BASELINE_ROOT/'input_plan.json'):
        raise ValueError('Historical physical-row input plan differs')
    qualifier = base_pair.qualification.configured_qualifier()
    for key, path in (('b17',STUDENT),('g4',TEACHER)):
        protected.update(qualifier.assets.verify_files(path,qualifier.assets.specifications()[key]['files']))
    summary = qualifier.old.read_sealed(base_pair.QUALIFICATION/'pair_summary.json')
    base_pair.validate_screen(summary, q.assets.sha256(base_pair.QUALIFICATION/'pair_summary.json'))
    paths.append(base_pair.QUALIFICATION/'pair_summary.json')
    for name in ('scripts/eval_math_batched.py','scripts/eval_qwen3_math_vllm.py',
                 'opd_ext/math_protocol.py','opd_ext/request_seeds.py',
                 'scripts/run_window_queue.py'):
        if q.assets.sha256(runtime/name) != q.assets.sha256(q.LOSS_REFERENCE/name):
            raise ValueError(f'Evaluation, prompt or sampling implementation drift: {name}')
    protected.update({str(p):q.assets.sha256(p) for p in paths})
    q.shared.verify_hashes(protected)
    q.jobs.write_json(RUN_ROOT/'baseline_alignment.json',dict(passed=True,
        baseline_root=str(BASELINE_ROOT),baseline_commit=BASELINE_COMMIT,same_input_plan=True,
        same_model_bytes=True,matched_environment_fields=['python','torch','transformers','vllm','tokenizers'],
        request_seed_rule='legacy',seed=21,
        comparison='Whole generation protocol, not a prompt-only causal claim',
        historical_training_protocol='qwen3_historical17_v1',historical_evaluation_protocol='legacy',
        new_training_and_evaluation_protocol=QWEN_COMPLETION_PROTOCOL,
        teacher_screen_status='inconclusive',training_authorized_despite_screen=True))
    return protected


def grading_gate(run):
    q = run.q
    digest = q.shared.grading.HISTORICAL_GRADER_SHA256
    q.shared.grading.validate_grader_hash(q.base.GRADER,digest)
    return dict(passed=True,grader_sha256=digest,scope='Exact historical grader bytes; no new grader')


def gpu_smoke(run, root, label):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    q = run.q
    model = STUDENT if label == 'student' else TEACHER
    tokenizer = AutoTokenizer.from_pretrained(model,local_files_only=True)
    selected = q.base.read_json(base_pair.QUALIFICATION/'selected.json')[:4]
    prompts = [completion_input_ids(tokenizer,r['question']) for r in selected]
    if tokenizer.eos_token_id != 151643:
        raise ValueError('Expected native Base EOS151643')
    folder = root/'gpu_smoke'/label
    folder.mkdir(parents=True,exist_ok=False)
    llm = LLM(model=str(model),dtype='bfloat16',tensor_parallel_size=1,
              gpu_memory_utilization=.6,max_model_len=18432,seed=21)
    sampling = SamplingParams(n=1,temperature=1.,top_p=.9,top_k=-1,max_tokens=256,
                              seed=21,ignore_eos=False,stop_token_ids=[])
    outputs = llm.generate([{'prompt_token_ids':ids} for ids in prompts],sampling,use_tqdm=False)
    rows = [dict(question=r['question'],prompt_token_ids=ids,response_token_ids=list(o.outputs[0].token_ids),
                 response_text=tokenizer.decode(o.outputs[0].token_ids,skip_special_tokens=False),
                 finish_reason=o.outputs[0].finish_reason,stop_reason=o.outputs[0].stop_reason)
            for r,ids,o in zip(selected,prompts,outputs)]
    q.jobs.write_json(folder/'raw.json',rows)
    if (len(rows) != 4 or any(o.prompt_token_ids != ids for o,ids in zip(outputs,prompts))
            or any(generated_think_tags(r['response_token_ids'],tokenizer=tokenizer,text=r['response_text']) for r in rows)):
        raise ValueError('Actual GPU completion input/nonthinking gate failed')
    q.jobs.write_json(folder/'acceptance.json',dict(passed=True,num_rollouts=4,chat_template_used=False,
        enable_thinking=False,eos_token_id=151643,stop_token_ids=[],generated_think_tags=0))


def write_comparison(q, root, results):
    original.n1.write_comparison(q,root,results)
    old = q.base.read_json(BASELINE_ROOT/'per_benchmark_results.json')
    comparisons = {}
    for task in q.shared.TASK_COUNTS:
        comparisons[task] = {str(step):{metric:dict(
            old_block_minus_token_pp=100*(old[task][f'block3_mean_step{step}'][metric]-old[task][f'token_opd_step{step}'][metric]),
            new_block_minus_token_pp=100*(results[f'block3_mean_step{step}']['per_task'][task][metric]-results[f'token_opd_step{step}']['per_task'][task][metric]))
            for metric in ('avg_at_8','pass_at_8')} for step in original.STEPS}
    q.jobs.write_json(root/'protocol_comparison.json',dict(baseline_root=str(BASELINE_ROOT),
        primary_checkpoint=100,single_training_seed=True,per_benchmark=comparisons,
        limitation='Whole-protocol association; not a prompt-only causal attribution'))


def configured_runner():
    run = base_pair.qualification.private_module('_ml2_base17_protocol_queue',original.__file__)
    run.q = base_pair.configured_runner()
    run.q.base = base_pair.qualification.private_module('_ml2_base17_protocol_base',run.q.base.__file__)
    run.ROOT,run.RUN_ROOT,run.HOST = ROOT,RUN_ROOT,'di-20260407234928-vrvxk'
    run.VENV,run.CACHE = run.q.base.VENV,Path('/dev/shm/mp17')
    run.PROTOCOL = QWEN_COMPLETION_PROTOCOL
    run.HARDWARE,run.LIFETIME = '4xA100-80GB; 64 Ray CPUs','Detached from SSH; not immune to host/platform termination'
    run.DISPLAY_PREFIX = 'ml2 Base1.7 GRPO4 protocol n1'
    run.ENTRYPOINT = Path(__file__).resolve()
    run.configure = partial(configure,run)
    run.validate_location = validate_location
    run.training_env,run.expected_card = partial(training_env,run),partial(expected_card,run)
    run.validate_rollout = validate_rollout
    run.preflight,run.grading_gate = partial(preflight,run),partial(grading_gate,run)
    run.gpu_smoke,run.write_comparison = partial(gpu_smoke,run),write_comparison
    run.LOSS_OVERRIDES = run.q.base.read_json(run.ENTRYPOINT.parents[1]/'configs/acp_component_source_hashes.json')
    return run


if __name__ == '__main__':
    configured_runner().main()
