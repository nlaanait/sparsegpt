"""Screen on 128 matched validation problems, then continue in the same process.

Keep the original evaluator and its immutable configuration unchanged so existing
answer records and complete 512-problem reports remain reusable.
"""
from __future__ import annotations

import argparse
import hashlib
from fractions import Fraction
import json
from pathlib import Path
import tempfile

from sparsifier.data.gsm8k import digest, load_manifest, write_immutable
from sparsifier.eval.gsm8k import read_completed
from sparsifier.eval import gsm8k_original

SCREEN_SIZE = 128
MAX_DROP = Fraction(3, 20)


class ScreenRejected(Exception):
    pass


def prepare_screen(manifest_path, reference, output, max_new_tokens):
    manifest = load_manifest(manifest_path)
    rows = manifest['rows']
    if len(rows) != 512 or manifest['source_split'] != 'train' or any(r['split'] != 'validation' for r in rows):
        raise ValueError('screen requires the complete 512-problem training validation manifest')
    ids = [r['id'] for r in rows]
    if len(set(ids)) != 512:
        raise ValueError('duplicate validation IDs')
    config = json.loads((reference/'config.json').read_text())
    report = json.loads((reference/'report.json').read_text())
    cfg_hash = digest(config)
    if (not report.get('complete') or report.get('config_sha256') != cfg_hash or
        report.get('behavior', {}).get('count') != 512 or config.get('arm') != 'dense' or
        config.get('model') != gsm8k_original.MODEL or config.get('revision') != gsm8k_original.REVISION or
        config.get('manifest_sha256') != manifest['manifest_sha256'] or
        config.get('evaluation_ids') != ids or config.get('max_new_tokens') != max_new_tokens):
        raise ValueError('dense reference does not match the full validation protocol')
    baseline = read_completed(reference/'answers.jsonl', cfg_hash)
    if set(baseline) != set(ids):
        raise ValueError('dense reference answers do not match the validation manifest')
    return rows, baseline, cfg_hash


def decision(rows, baseline, done, baseline_hash, config_hash):
    ids = [row['id'] for row in rows[:SCREEN_SIZE]]
    if len(ids) != SCREEN_SIZE or not set(ids).issubset(done):
        return None
    dense_correct = sum(bool(baseline[i]['correct']) for i in ids)
    candidate_correct = sum(bool(done[i]['correct']) for i in ids)
    passed = Fraction(dense_correct - candidate_correct, SCREEN_SIZE) <= MAX_DROP
    return {'schema':1, 'purpose':'exploratory_screen_not_retention_claim',
            'screen_count':SCREEN_SIZE, 'screen_ids_sha256':digest(ids),
            'dense_config_sha256':baseline_hash, 'candidate_config_sha256':config_hash,
            'max_accuracy_drop_pp':15, 'dense_correct':dense_correct,
            'candidate_correct':candidate_correct, 'accuracy_drop_pp':100*(dense_correct-candidate_correct)/SCREEN_SIZE,
            'passed':passed, 'screen_decision_complete':True}


def screened_behavior(original_behavior, rows, baseline, baseline_hash, model, tokenizer, output, config_hash, *, max_new_tokens):
    path = output/'answers.jsonl'
    done = read_completed(path, config_hash)
    ids = {row['id'] for row in rows}
    if set(done) - ids:
        raise ValueError('unexpected result IDs')
    # Reuse the exact existing evaluator for each missing screen example. Append
    # each answer immediately, so an interruption costs at most one example.
    with tempfile.TemporaryDirectory(prefix='screen-', dir=output) as tmp:
        temp = Path(tmp)
        for row in rows[:SCREEN_SIZE]:
            if row['id'] in done:
                continue
            original_behavior(model, tokenizer, [row], temp, config_hash, max_new_tokens=max_new_tokens)
            one = read_completed(temp/'answers.jsonl', config_hash)
            if set(one) != {row['id']}:
                raise ValueError('screen evaluator returned unexpected answers')
            record = one[row['id']]
            with path.open('a', buffering=1) as handle:
                handle.write(json.dumps(record)+'\n')
                handle.flush()
            done[row['id']] = record
            (temp/'answers.jsonl').unlink()
            print(json.dumps({'screen_evaluated':sum(r['id'] in done for r in rows[:SCREEN_SIZE]),
                              'screen_total':SCREEN_SIZE}), flush=True)
    result = decision(rows, baseline, done, baseline_hash, config_hash)
    if result is None:
        raise ValueError('screen incomplete')
    write_immutable(output/'screen.json', result)
    print(json.dumps({'screen': result}), flush=True)
    if not result['passed']:
        raise ScreenRejected()
    # The normal evaluator re-reads answers.jsonl and skips the screen answers.
    return original_behavior(model, tokenizer, rows, output, config_hash, max_new_tokens=max_new_tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-new-tokens',type=int,default=2048)
    parser.add_argument('--screen-reference',type=Path,required=True)
    args = parser.parse_args()
    rows, baseline, baseline_hash = prepare_screen(args.manifest,args.screen_reference,args.output,args.max_new_tokens)
    # Existing partial runs may already contain the whole screen. Check their
    # provenance and decide before loading a model or redoing diagnostics.
    cfg_path = args.output/'config.json'
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        source_cfg = json.loads((args.source/'config.json').read_text())
        source_report = json.loads((args.source/'report.json').read_text())
        expected_ids = [r['id'] for r in rows]
        if (cfg.get('model') != gsm8k_original.MODEL or cfg.get('revision') != gsm8k_original.REVISION or
            cfg.get('manifest_sha256') != json.loads((args.screen_reference/'config.json').read_text())['manifest_sha256'] or
            cfg.get('evaluation_ids') != expected_ids or cfg.get('max_new_tokens') != args.max_new_tokens or
            cfg.get('source_config_sha256') != digest(source_cfg) or not source_report.get('complete') or
            source_report.get('config_sha256') != digest(source_cfg) or
            cfg.get('candidate_chain') != [s['candidate_sha256'] for s in source_report['stages']] or
            cfg.get('source_sha256') != hashlib.sha256(Path(gsm8k_original.__file__).read_bytes()).hexdigest()):
            raise ValueError('existing candidate evaluation provenance mismatch')
        cfg_hash = digest(cfg)
        done = read_completed(args.output/'answers.jsonl', cfg_hash)
        if set(done) - set(expected_ids):
            raise ValueError('unexpected result IDs')
        result = decision(rows, baseline, done, baseline_hash, cfg_hash)
        if result is not None:
            write_immutable(args.output/'screen.json', result)
            print(json.dumps({'screen':result}),flush=True)
            if not result['passed']:
                print('Screen rejected: saved answers retained; no 512-problem result.',flush=True)
                return
    old_behavior = gsm8k_original.behavior
    def run_screen(model, tokenizer, actual_rows, output, config_hash, *, max_new_tokens):
        if [r['id'] for r in actual_rows] != [r['id'] for r in rows]:
            raise ValueError('validation rows changed after screen preparation')
        return screened_behavior(old_behavior,rows,baseline,baseline_hash,model,tokenizer,output,config_hash,max_new_tokens=max_new_tokens)
    gsm8k_original.behavior = run_screen
    import sys
    sys.argv = ['sparsifier.eval.gsm8k_original','--manifest',str(args.manifest),
                '--source',str(args.source),'--output',str(args.output),
                '--max-new-tokens',str(args.max_new_tokens)]
    try:
        gsm8k_original.main()
    except ScreenRejected:
        print('Screen rejected: no 512-problem result; saved answers remain available.',flush=True)
    finally:
        gsm8k_original.behavior = old_behavior

if __name__ == '__main__': main()
