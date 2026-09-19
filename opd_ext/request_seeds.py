"""Stable request-local randomness, independent of rank, UUIDs and launch order."""

from copy import deepcopy
import hashlib
import json

SEED_RULE = 'sha256_step_question_sample_v1'


def request_identities(sources, *, group_size, global_seed, step):
    if group_size < 1 or step < 1 or global_seed < 0:
        raise ValueError('Invalid request seed coordinates')
    result, seen = [], set()
    for source in sources:
        if 'index' not in source or not isinstance(source.get('question'), str):
            raise ValueError('Stable dataset index and question are required')
        index = int(source['index'])
        if index in seen:
            raise ValueError('Duplicate dataset index within a training batch')
        seen.add(index)
        digest = hashlib.sha256(source['question'].encode('utf-8')).hexdigest()
        result.extend({'rule': SEED_RULE, 'global_seed': int(global_seed), 'step': int(step),
                       'source_index': index, 'question_sha256': digest, 'sample_index': j}
                      for j in range(group_size))
    return result


def request_seed(identity, *, turn):
    if identity['rule'] != SEED_RULE or turn < 0:
        raise ValueError('Unknown request seed rule or invalid turn')
    payload = json.dumps({**identity, 'turn': int(turn)}, sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') & ((1 << 63) - 1)


def per_request_sampling(sampling, seeds):
    if sampling.n != 1:
        raise ValueError('Independent requests require n=1; expand replicas before sharding')
    result = []
    for seed in seeds:
        value = deepcopy(sampling)
        value.seed = int(seed)
        result.append(value)
    return result
