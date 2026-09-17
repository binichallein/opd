#!/usr/bin/env python3
"""Exercise the actual math environment/collector and vLLM non-thinking inputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opd_ext.math_protocol import (DISABLED_SUFFIX, PROTOCOL, STOP_TOKEN_IDS,
                                   LLAMA_PROTOCOL, LLAMA_STOP_IDS, valid_control_prefix, evaluation_inputs,
                                   generation_record, generated_think_tags, math_prompt, render_nonthinking)


def configure_gpu_process():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--eval-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', choices=(PROTOCOL, LLAMA_PROTOCOL), default=PROTOCOL)
    parser.add_argument('--cpu-only', action='store_true', help='Check the real collector inputs without loading any model')
    args=parser.parse_args()
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        configure_gpu_process()
    from transformers import AutoTokenizer
    import numpy as np
    from omegaconf import OmegaConf
    import torch
    from verl import DataProto
    from verl.utils.tokenizer import hf_tokenizer
    from agent_system.environments.env_manager import MathEnvironmentManager
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from eval_qwen3_math_vllm import apply_template

    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer=hf_tokenizer(args.model,local_files_only=True)
    assert hashlib.sha256(args.cohort.read_bytes()).hexdigest() == 'dd9e79bc95378ae3ac4cbb194b511f3fed7d1b6fb9050ac29ee8d93ea3f33aee'
    cohort=[json.loads(line) for line in args.cohort.read_text().splitlines()][:16]
    config=OmegaConf.create({'data':{'opd_prompt_protocol':args.protocol,
                            'apply_chat_template_kwargs':{'enable_thinking':False},
                            'max_prompt_length':2048,'truncation':'middle','return_raw_chat':True}})
    manager=MathEnvironmentManager.__new__(MathEnvironmentManager)
    manager.config=config
    manager.tasks=[row['question'] for row in cohort]
    observations=manager.build_text_obs(manager.tasks)
    collector=TrajectoryCollector(config,tokenizer)
    prompts=[]
    for i,row in enumerate(cohort):
        raw_prompt=np.empty(1,dtype=object)
        raw_prompt[0]=[{'role':'user','content':row['question']}]
        gen=DataProto.from_single_dict({'input_ids':torch.zeros((1,1),dtype=torch.long),
                                       'raw_prompt':raw_prompt,'data_source':np.array(['dapo-math-17k'],dtype=object)})
        processed=collector.preprocess_single_sample(0,gen,{'text':[observations[i]]})
        evaluation=apply_template(tokenizer,math_prompt(row['question']),False)
        assert evaluation==render_nonthinking(tokenizer,observations[i])
        assert valid_control_prefix(evaluation, tokenizer, args.protocol)
        ids=tokenizer.encode(evaluation,add_special_tokens=False)
        if args.protocol == LLAMA_PROTOCOL:
            assert evaluation_inputs(tokenizer, [evaluation])[0]['prompt_token_ids'] == ids
        if len(ids)>2048:
            ids=ids[:1024]+ids[-1024:]
        assert processed['raw_prompt_ids']==ids
        assert ids==processed['input_ids'][processed['attention_mask'].bool()].tolist()
        prompts.append({'index':row['index'],'question':row['question'],'answer':row['answer'],
                        'prompt_text':evaluation,'prompt_token_ids':ids})
    checked={}
    for file in sorted(args.eval_data.glob('*.jsonl')):
        rows=[json.loads(line) for line in file.read_text().splitlines()]
        for row in rows:
            assert math_prompt(row['problem'])==row['prompt'], (file,row['id'])
        checked[file.name]=len(rows)
    with (args.output/'input_contract.json').open('x') as f:
        json.dump({'protocol':args.protocol,'enable_thinking':False,'eval_rows_checked':checked,
                   'vllm_worker_multiproc_method':os.environ['VLLM_WORKER_MULTIPROC_METHOD'],
                   'train_eval_prompt_ids_identical':True,'n':len(prompts)},f,indent=2)

    if args.cpu_only:
        print(json.dumps({'cpu_input_contract_passed': True, 'gpu_check_performed': False}), flush=True)
        return
    from vllm import LLM, SamplingParams
    llm=LLM(model=args.model,tokenizer=args.model,dtype='bfloat16',tensor_parallel_size=1,
            gpu_memory_utilization=.6,max_model_len=18432,max_num_batched_tokens=18432,
            max_num_seqs=32,enforce_eager=False,enable_chunked_prefill=False,seed=21)
    params=SamplingParams(n=1,temperature=1.,top_p=.9,top_k=-1,seed=21,max_tokens=16384,
                          ignore_eos=False,stop_token_ids=LLAMA_STOP_IDS if args.protocol == LLAMA_PROTOCOL else STOP_TOKEN_IDS,
                          detokenize=False,logprobs=0)
    for i,row in enumerate(prompts):
        llm.llm_engine.add_request(str(i),{'prompt_token_ids':row['prompt_token_ids']},params)
    records=[]
    with (args.output/'raw.jsonl').open('x') as f:
        while llm.llm_engine.has_unfinished_requests():
            for result in llm.llm_engine.step():
                if not result.finished:
                    continue
                row=prompts[int(result.request_id)]
                assert list(result.prompt_token_ids) == row['prompt_token_ids']
                record={**row,**generation_record(row['prompt_token_ids'],result.outputs[0],params,tokenizer.eos_token_id)}
                record['response_text']=tokenizer.decode(record['response_token_ids'],skip_special_tokens=False,
                                                         clean_up_tokenization_spaces=False)
                record['generated_think_tags']=(generated_think_tags(record['response_token_ids'])
                                               or '<think>' in record['response_text'] or '</think>' in record['response_text'])
                f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                f.flush()
                os.fsync(f.fileno())
                records.append(record)
                print(json.dumps({'completed':len(records),'index':row['index'],
                                  'tokens':len(record['response_token_ids']),
                                  'finish':record['finish_reason'],'generated_think_tags':record['generated_think_tags']}),flush=True)
    violations=sum(r['generated_think_tags'] for r in records)
    summary={'passed':len(records)==16 and violations==0,'protocol':args.protocol,'n':len(records),
             'generated_think_tag_count':violations,'length_stop_count':sum(r['finish_reason']=='length' for r in records),
             'mean_length':sum(len(r['response_token_ids']) for r in records)/len(records),
             'raw_sha256':hashlib.sha256((args.output/'raw.jsonl').read_bytes()).hexdigest(),
             'limitation':'Sampled GPU check; not a guarantee that Base will always obey control tokens.'}
    with (args.output/'summary.json').open('x') as f:
        json.dump(summary,f,indent=2)
    print(json.dumps(summary),flush=True)
    if not summary['passed']:
        raise RuntimeError('GPU non-thinking gate failed; preserve outputs and do not start training')


if __name__=='__main__':
    main()
