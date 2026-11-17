"""Render verified, paired EOS effects without fitting a sparsity frontier."""
from __future__ import annotations

import argparse
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path

import plotly.graph_objects as go

PAIRS = [('sparsegpt-task-s55', 'SparseGPT 55%'), ('sparsegpt-task-s60', 'SparseGPT 60%'),
         ('local-s55', 'Local 55%'), ('local-s70', 'Local 70%'),
         ('downstream-weightdiag-s55', 'Downstream 55%'),
         ('downstream-weightdiag-s70', 'Downstream 70%')]


def load(path):
    return json.loads(path.read_text())


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/tangent/gsm8k/eos-corrected-v1'))
    args = parser.parse_args(); root = args.root.resolve(); audits = root/'audits'
    dense = load(audits/'dense.json'); assert dense['common128_complete'] and dense['count'] == 128
    points = []
    for label, name in PAIRS:
        cp = audits/f'{label}-historical-comparison.json'; ap = audits/f'{label}.json'
        lp = audits/f'{label}-checkpoint-score-link.json'
        comparison, audit, link = map(load, (cp, ap, lp))
        assert comparison['count'] == 128 and comparison['decoded_prefix_matches'] == 128
        assert comparison['max_answer_nll_abs_difference'] == 0.
        assert comparison['ids_sha256'] == audit['common128']['ids_sha256'] == dense['common128']['ids_sha256']
        assert audit['common128_complete'] and audit['raw_token_rescore_mismatches'] == 0
        assert link['verified_stage_count'] == 96 and link['correct'] == comparison['corrected_correct']
        assert comparison['corrected_config_sha256'] == audit['config_sha256'] == link['evaluation_config_sha256']
        assert link['raw_audit_sha256'] == checksum(ap)
        points.append({'label': label, 'name': name, 'count': 128,
            'old_correct': comparison['old_correct'], 'corrected_correct': comparison['corrected_correct'],
            'old_truncated': comparison['old_truncated'], 'corrected_truncated': comparison['corrected_truncated'],
            'comparison_sha256': checksum(cp), 'raw_audit_sha256': checksum(ap),
            'checkpoint_link_sha256': checksum(lp)})
    figure = go.Figure()
    for prefix, title, color in [('old', 'Historical: model EOS only', '#a6a9ae'),
                                 ('corrected', 'Corrected: model and tokenizer EOS', '#146f9b')]:
        figure.add_trace(go.Bar(name=title, orientation='h', y=[p['name'] for p in points],
            x=[100*p[f'{prefix}_correct']/128 for p in points],
            text=[f"{p[f'{prefix}_correct']}/128" for p in points], textposition='outside',
            cliponaxis=False, marker_color=color,
            customdata=[[p[f'{prefix}_correct'],p[f'{prefix}_truncated']] for p in points],
            hovertemplate='%{y}<br>Correct: %{customdata[0]}/128 (%{x:.2f}%)<br>Truncated: %{customdata[1]}/128<extra>%{fullData.name}</extra>'))
    figure.add_vline(x=100*dense['correct']/128, line_dash='dash', line_color='#555',
                    annotation_text=f"Dense {dense['correct']}/128", annotation_position='top left')
    figure.update_layout(template='plotly_white', width=1100, height=640, barmode='group',
        title={'text':'EOS omission distorted GSM8K pruning scores<br><sup>Qwen3.5-9B | seed 0 | batch 32 | 2,048 new tokens | common 128 questions</sup>', 'x':.025},
        margin=dict(l=160,r=75,t=135,b=95),
        legend=dict(orientation='h',x=0,y=1.12),
        xaxis=dict(title='Correct answers (% of common 128)',range=[0,105],tickvals=list(range(0,101,20))),
        yaxis=dict(autorange='reversed'),
        annotations=[dict(x=0,y=-.18,xref='paper',yref='paper',showarrow=False,xanchor='left',
            text='Eligible MLP sparsity. Each pair uses one frozen checkpoint; parser and budget are unchanged.<br>Exploratory validation. The incomplete sparsity grid does not establish a zero-accuracy threshold.',
            font=dict(size=11,color='#555'))]+list(figure.layout.annotations))
    out = root/'figures'; out.mkdir(exist_ok=True)
    html_path = out/'eos-effect.html'; figure.write_html(html_path, include_plotlyjs=True, full_html=True)
    class Scripts(HTMLParser):
        def __init__(self):super().__init__();self.remote=[]
        def handle_starttag(self,tag,attrs):
            if tag=='script' and dict(attrs).get('src'):self.remote.append(dict(attrs)['src'])
    parsed=Scripts();parsed.feed(html_path.read_text());assert not parsed.remote
    data={'scope':'paired_EOS_artifact_effect_not_complete_sparsity_curve','model':'Qwen/Qwen3.5-9B',
        'common128_ids_sha256':dense['common128']['ids_sha256'], 'dense_correct':dense['correct'],
        'dense_raw_audit_sha256':checksum(audits/'dense.json'), 'points':points,
        'generator_sha256':checksum(Path(__file__)), 'html_sha256':checksum(html_path),
        'external_script_dependencies':parsed.remote}
    (out/'eos-effect.data.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'html':str(html_path),'paired_conditions':len(points),'external_scripts':parsed.remote}))


if __name__=='__main__':main()
