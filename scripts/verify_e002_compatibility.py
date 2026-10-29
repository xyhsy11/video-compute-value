"""Independent read-only replay of the bounded E2 probe. Writes only an E2 receipt."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from video_compute_value.common import digest, file_hash, read_json, read_jsonl, write_json
from video_compute_value.protocol import parse_answer


def verify(root, run):
    receipt = read_json(run/'compatibility.json')
    assert receipt['worker_reaped'] and receipt['automatic_retries'] == 0
    events = read_jsonl(run/'private/compatibility_calls.jsonl')
    completed = [e for e in events if e['event'] == 'completed']
    assert len(completed) == receipt['completed_model_calls']
    rows = [read_json(p) for p in sorted((run/'private/compatibility_records').glob('*.json'))]
    assert len(rows) == len(completed)
    frozen = read_json(root/'results/e001_20261029_132707_v1/private/frozen.json')
    tokenizer = Tokenizer.from_file(str(Path(frozen['model_path'])/'tokenizer.json'))
    for row in rows:
        decoded = tokenizer.decode(row['generated_token_ids'], skip_special_tokens=True)
        assert decoded == row['decoded_output'] and parse_answer(decoded) == row['parsed_answer']
        assert len(row['generated_token_ids']) == row['output_tokens'] <= 8
        assert row['generation_policy_sha256'] == digest(read_json(run/'private/compatibility_worker.json')['generation_policy'])
        assert row['frames_selected'] in (8, 16)
        assert row['frames_selected'] == row['frames_processor'] == row['frames_represented']
        assert row['spatial_height'] == 192 and row['spatial_width'] == 320
    integrity = {r['condition']: r for r in rows if r['kind'] == 'integrity'}
    capture_verified = False
    if receipt['status'] == 'PASSED':
        assert len(rows) == 5
        plain, observed, quarter = (integrity[name] for name in ('base_plain', 'base_observed', 'quarter1'))
        for key in ('generated_token_ids', 'input_tensor_hashes', 'actual_timestamps', 'source_indices',
                    'retained_base_rgb_hashes', 'generation_policy_sha256'):
            assert plain[key] == observed[key]
        assert quarter['retained_base_rgb_hashes'] == plain['retained_base_rgb_hashes']
        assert quarter['visual_tokens'] == plain['visual_tokens']*2
        c = observed['C']
        assert len(c) == 3 and all(0 <= v <= 1 for v in c)
        assert set(observed['features']) == {'V', 'Q', 'VQ'}
        assert [len(observed['features'][k]) for k in ('V', 'Q', 'VQ')] == [19, 66, 341]
        assert all(np.isfinite(v).all() for v in observed['features'].values())
        capture_verified = True
    result = {'checks_passed': True, 'real_saved_calls_checked': len(rows),
        'plain_observed_token_agreement': capture_verified, 'native_8_16_checked': capture_verified,
        'same_call_generation_scores_present': capture_verified,
        'first_token_full_scores_retained': False,
        'limitation': 'One legacy input compatibility check; not 120-question integrity or timing repeatability.',
        'compatibility_receipt_sha256': file_hash(run/'compatibility.json')}
    write_json(run/'compatibility_verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    print(verify(Path(__file__).resolve().parents[1], args.run_dir.resolve()))
