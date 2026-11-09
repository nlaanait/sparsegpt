"""Dense-relative retention bounds and measured sparsity frontiers.

Fractions, not percentage points, are used in the API. Bounds are conservative;
zero observed disagreements in a small sample does not establish equivalence.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import beta
from sparsifier.data.gsm8k import digest
from sparsifier.eval.gsm8k import atomic_json, read_completed


def retention(candidate, dense, *, margin, unit='family', confidence=.95):
    if not 0 <= margin <= 1 or not 0 < confidence < 1:
        raise ValueError('invalid tolerance or confidence')
    if not candidate or set(candidate) != set(dense):
        raise ValueError('identical nonempty evaluation IDs required')
    losses = gains = dense_correct = 0
    groups = {}
    for key in sorted(candidate):
        a,b = candidate[key],dense[key]
        if a['expected_answer'] != b['expected_answer'] or a['family'] != b['family']:
            raise ValueError('paired target/family mismatch')
        lost = bool(b['correct']) and not bool(a['correct'])
        gained = bool(a['correct']) and not bool(b['correct'])
        losses += lost; gains += gained; dense_correct += bool(b['correct'])
        groups.setdefault(a['family'],[]).append(int(lost)-int(gained))
    n = len(candidate); drop = (losses-gains)/n; tail = 1-confidence
    if unit == 'problem':
        # Union bound over exact marginal binomial bounds remains valid for
        # the dependent loss/gain counts from the same paired outcomes.
        loss_upper = 1. if losses == n else float(beta.ppf(1-tail/2,losses+1,n-losses))
        gain_lower = 0. if gains == 0 else float(beta.ppf(tail/2,gains,n-gains+1))
        upper = loss_upper-gain_lower
        method = 'Bonferroni_exact_binomial_loss_minus_gain'
        assumption = 'independent representative problems; fixed candidate'
    elif unit == 'family':
        # Bounded family means in [-1,1], arbitrary dependence within a family.
        weights = np.array([len(v)/n for v in groups.values()])
        upper = min(1.,drop+math.sqrt(2*float(weights@weights)*math.log(1/tail)))
        method = 'weighted_family_Hoeffding'
        assumption = 'independent representative families; fixed candidate and family weights'
    else:
        raise ValueError('unit must be problem or family')
    return {'problems':n,'families':len(groups),'dense_accuracy':dense_correct/n,
            'candidate_accuracy':(dense_correct-losses+gains)/n,'accuracy_drop':drop,
            'dense_correct_lost':losses,'dense_wrong_recovered':gains,
            'dense_correct_retained':dense_correct-losses,'allowed_drop':margin,
            'confidence':confidence,'drop_upper_bound':upper,
            'point_estimate_within_tolerance':drop <= margin,
            'retention_established_under_assumptions':upper <= margin,
            'bound_method':method,'assumptions':assumption}


def frontier(points):
    """No interpolation, monotonicity assumption, or missing-count substitution."""
    measured = [p for p in points if p.get('whole_model_removed_fraction') is not None]
    def best(field):
        valid = [p for p in measured if p['retention'][field]]
        return max(valid,key=lambda p:p['whole_model_removed_fraction'])['run'] if valid else None
    return {'largest_observed_within_tolerance':best('point_estimate_within_tolerance'),
            'largest_statistically_supported':best('retention_established_under_assumptions')}


def load_run(path):
    config = json.loads((path/'config.json').read_text())
    report = json.loads((path/'report.json').read_text())
    if not report.get('complete') or report['config_sha256'] != digest(config):
        raise ValueError(f'incomplete or mismatched run: {path}')
    answers = read_completed(path/'answers.jsonl',digest(config))
    if set(answers) != set(config['evaluation_ids']):
        raise ValueError('evaluation coverage mismatch')
    return config,report,answers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dense',type=Path,required=True)
    p.add_argument('--candidate',type=Path,action='append',required=True)
    p.add_argument('--allowed-drop',type=float,required=True,help='fraction, e.g. .02 for two points')
    p.add_argument('--unit',choices=['family','problem'],default='family')
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    dc,dr,da = load_run(args.dense)
    if dc.get('arm') != 'dense': raise ValueError('dense reference must be a dense run')
    if len(set(args.candidate)) != len(args.candidate): raise ValueError('duplicate candidates')
    points = []
    # Simultaneous coverage across this explicitly supplied candidate collection.
    confidence = 1-.05/len(args.candidate)
    for path in args.candidate:
        c,r,a = load_run(path)
        for key in ('model','revision','manifest_sha256','max_new_tokens'):
            if c.get(key) is None or c[key] != dc.get(key):
                raise ValueError(f'protocol mismatch: {key}')
        result = retention(a,da,margin=args.allowed_drop,unit=args.unit,confidence=confidence)
        points.append({'run':str(path),'arm':c['arm'],
                       'whole_model_removed_fraction':r.get('whole_model_removed_fraction'),
                       'retention':result})
    by_arm = {}
    for point in points: by_arm.setdefault(point['arm'],[]).append(point)
    result = {'dense':str(args.dense),'allowed_drop':args.allowed_drop,'points':points,
              'frontiers':{arm:frontier(rows) for arm,rows in by_arm.items()},
              'scope':'development_selection_only_not_test_confirmation',
              'multiplicity':'Bonferroni across supplied candidates; includes both binomial tails',
              'limitations':['candidate list must include all candidates used for selection',
                             'bounds rely on representative independent sampling units',
                             'unreported whole-model counts excluded from frontier',
                             'same model weights with different generation protocols cannot be pooled']}
    atomic_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
