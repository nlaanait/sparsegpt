import json
from pathlib import Path

from sparsifier.data.gsm8k import load_manifest
from sparsifier.eval import gsm8k_decision, gsm8k_decision_full as full

ROOT = Path(__file__).resolve().parents[1] / 'results/tangent/gsm8k/data'


def test_screen_reuses_existing_reference_order():
    reserved = load_manifest(ROOT/'original-validation-v1.json')
    calibration = load_manifest(ROOT/'controlled-v1.json')
    manifest = full.prepare_manifest(reserved, calibration)
    old = gsm8k_decision.prepare_manifest(reserved, calibration)
    full.validate_protocol(manifest)
    assert manifest['rows'][:128] == old['rows']
    assert manifest['reference_manifest_sha256'] == old['manifest_sha256']
    assert len(manifest['rows']) == 512


def test_128_screen_continues_without_repeating_answers(tmp_path):
    rows = [{'id': str(i)} for i in range(512)]
    manifest = {'rows': rows, 'screen_ids_sha256': 'test'}
    baseline = {str(i): {'correct': True} for i in range(128)}
    calls = []
    def evaluate(model, tokenizer, group, output, config_hash, **kwargs):
        calls.extend(row['id'] for row in group)
        with (output/'answers.jsonl').open('w') as stream:
            for row in group:
                stream.write(json.dumps({'id': row['id'], 'config_sha256': config_hash,
                                         'correct': True})+'\n')
    done, decision = full.run_tiers(evaluate, None, None, manifest, tmp_path, 'hash', baseline=baseline)
    assert decision['passed'] and len(done) == 512 and len(calls) == 512
    assert len(set(calls)) == 512
    full.run_tiers(evaluate, None, None, manifest, tmp_path, 'hash', baseline=baseline)
    assert len(calls) == 512


def test_128_screen_stops_rejected_candidate(tmp_path):
    rows = [{'id': str(i)} for i in range(512)]
    manifest = {'rows': rows, 'screen_ids_sha256': 'test'}
    baseline = {str(i): {'correct': True} for i in range(128)}
    def evaluate(model, tokenizer, group, output, config_hash, **kwargs):
        with (output/'answers.jsonl').open('w') as stream:
            for row in group:
                stream.write(json.dumps({'id': row['id'], 'config_sha256': config_hash,
                                         'correct': False})+'\n')
    done, decision = full.run_tiers(evaluate, None, None, manifest, tmp_path, 'hash', baseline=baseline)
    assert not decision['passed'] and len(done) == 128
