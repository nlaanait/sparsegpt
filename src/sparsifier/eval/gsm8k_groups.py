"""Matched intervention contrasts with shared-problem uncertainty (CPU only)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sparsifier.data.gsm8k import digest, load_manifest
from sparsifier.eval.gsm8k import atomic_json, read_completed
from sparsifier.eval.gsm8k_prediction import crossed_interval


def matched_rows(report, config):
    groups = {}
    for row in report['outcomes']:
        if row['kind'] == 'representative': continue
        key = (row['match_id'],row['compensation'],row['kind'])
        if key in groups: raise ValueError('duplicate matched outcome')
        groups[key] = row
    expected = {(i,mode,kind) for i in range(config['group_pairs'])
                for mode in ('none','obs') for kind in ('high_geometry','low_geometry')}
    if not set(groups) <= expected: raise ValueError('unexpected matched outcome')
    if set(groups) != expected: return None
    used = set()
    for i in range(config['group_pairs']):
        for kind in ('high_geometry','low_geometry'):
            a,b = groups[i,'none',kind],groups[i,'obs',kind]
            if a['entries'] != b['entries']: raise ValueError('compensation changes intervention membership')
            entries = set(a['entries'])
            if len(entries) != config['group_size'] or used & entries:
                raise ValueError('groups must be disjoint with declared size')
            used.update(entries)
        high,low = groups[i,'none','high_geometry'],groups[i,'none','low_geometry']
        ratio = max(high['ambient_cost_sum'],low['ambient_cost_sum']) / max(min(high['ambient_cost_sum'],low['ambient_cost_sum']),1e-30)
        if ratio > 1.100001: raise ValueError('ambient matching tolerance violated')
    return groups


def summarize_matrix(values,families):
    values = np.asarray(values,dtype=float)
    if values.ndim != 2 or values.shape[1] != len(families) or not np.isfinite(values).all():
        raise ValueError('invalid paired outcome matrix')
    return {'mean':float(values.mean()),'exploratory_crossed_ci95':crossed_interval(values,families),
            'per_match_mean':values.mean(1).tolist()}


def analyze(run,report,config,manifest):
    if report['config_sha256'] != digest(config) or config['manifest_sha256'] != manifest['manifest_sha256']:
        raise ValueError('provenance mismatch')
    groups = matched_rows(report,config)
    result = {'scope':'development_matched_interventions','config_sha256':digest(config),
              'complete':bool(report.get('complete')) and groups is not None,
              'completed_group_conditions':sum(r['kind']!='representative' for r in report['outcomes'])}
    if not result['complete']: return result
    ids = config['evaluation_ids']; rows = {r['id']:r for r in manifest['rows']}
    families = [rows[i]['family'] for i in ids]
    result.update(problem_count=len(ids),source_families=len(set(families)),
        matched_pairs=config['group_pairs'],no_op_max_nll_difference=report['no_op_max_nll_difference'],
        limitations=['exploratory, no multiplicity correction','few independent families and matched groups',
                     'bootstrap zero-width intervals do not establish equivalence',
                     'summed isolated scores describe selection only; behavioral contrasts use actual joint interventions'])
    for mode in ('none','obs'):
        nll,accuracy,ambient,projected = [],[],[],[]
        for i in range(config['group_pairs']):
            high,low = groups[i,mode,'high_geometry'],groups[i,mode,'low_geometry']
            nll.append(np.array(high['answer_nll'])-np.array(low['answer_nll']))
            outcomes = []
            for row in (high,low):
                answers = read_completed(run/(row['id']+'-'+mode)/'answers.jsonl',
                                         digest([digest(config),row['id'],mode]))
                if set(answers) != set(ids): raise ValueError('incomplete generated answers')
                for identifier in ids:
                    if answers[identifier]['expected_answer'] != rows[identifier]['answer']:
                        raise ValueError('answer target mismatch')
                outcomes.append(np.array([int(answers[j]['correct']) for j in ids]))
            accuracy.append(outcomes[0]-outcomes[1])
            ambient.append(high['local']['ambient_readout_mse']-low['local']['ambient_readout_mse'])
            projected.append(high['local']['heldout_local_span_mse']-low['local']['heldout_local_span_mse'])
        result[mode] = {'high_minus_low_answer_nll':summarize_matrix(nll,families),
                        'high_minus_low_accuracy':summarize_matrix(accuracy,families),
                        'per_match_ambient_mse_difference':ambient,
                        'per_match_projected_mse_difference':projected}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    args = p.parse_args()
    result = analyze(args.run,json.loads((args.run/'report.json').read_text()),
                     json.loads((args.run/'config.json').read_text()),load_manifest(args.manifest))
    atomic_json(args.run/'matched-analysis.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
