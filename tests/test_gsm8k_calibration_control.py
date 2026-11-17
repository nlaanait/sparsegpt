"""Contracts for the isolated calibration ablation adapter."""
import json
import sys

import numpy as np
import pytest
import torch

from scripts import run_gsm8k_calibration_control as control
from sparsifier.data.gsm8k import digest,write_immutable
from sparsifier.eval.gsm8k_batching import pack


class Tokenizer:
    pad_token_id = 0


def rows():
    result = []
    for n,(prefix,solution) in enumerate([([1,2,3,4],[5,6,7]),([2,3,4,5,6,7],[8,9,10,11,12])]):
        tokens = prefix+solution+[13]
        result.append({'id':str(n),'source_sha256':str(n),'split':'geometry_fit',
            'input_ids':tokens,'input_ids_sha256':digest(tokens),'prompt_tokens':len(prefix),
            'response_start':len(prefix),'prompt_readout_position':len(prefix)-1})
    return result


def bundle():
    return {'protocol':'gsm8k-reasoning-calibration-control-v1','rows':rows(),'eos_token_id':13,
            'calibration_manifest_sha256':'parents'}


@pytest.mark.parametrize('anchor_type',['linear_attention','full_attention'])
def test_reasoning_capture_keeps_causal_prompt_readouts_and_excludes_padding(anchor_type,monkeypatch):
    from sparsifier.eval.gsm8k_batching import reference_backend
    from sparsifier.eval.gsm8k_capture import capture_one
    from sparsifier.models.qwen import _PYTORCH_ATTENTION
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    for name in _PYTORCH_ATTENTION:
        monkeypatch.setattr(modeling,name,getattr(modeling,name))
    reference_backend()
    config = Qwen3_5TextConfig(vocab_size=32,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        linear_key_head_dim=8,linear_value_head_dim=8,linear_num_key_heads=1,linear_num_value_heads=2,
        layer_types=['linear_attention',anchor_type],
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':.5})
    config._attn_implementation = 'sdpa'
    torch.manual_seed(704)
    model = modeling.Qwen3_5TextModel(config).eval().requires_grad_(False)
    target = model.layers[1].mlp.down_proj
    selected = rows()
    inputs = pack([r['input_ids'] for r in selected],Tokenizer(),'cpu')
    actual = control.capture_control(model,target,inputs,selected,'reasoning')
    for row,record in zip(selected,actual):
        prefix = row['input_ids'][:row['prompt_tokens']]
        prompt = capture_one(model,target,torch.tensor([prefix]))
        np.testing.assert_allclose(record['x'],prompt['x'],rtol=5e-4,atol=1e-5)
        np.testing.assert_allclose(record['y'],prompt['y'],rtol=5e-4,atol=1e-5)
        complete = capture_one(model,target,torch.tensor([row['input_ids']]),token_samples=len(row['input_ids']))
        np.testing.assert_allclose(record['pool'],complete['token_x'][len(prefix):-2],rtol=5e-4,atol=1e-5)
        np.testing.assert_allclose(record['terminal'],complete['token_x'][-2],rtol=5e-4,atol=1e-5)
        assert len(record['pool']) == len(row['input_ids'])-len(prefix)-2
    assert not target._forward_hooks


def test_equal_budget_collectors_retain_readouts_and_every_answer_endpoint():
    from test_gsm8k_batching import Model
    model = Model().eval()
    c4 = np.array([[1,2,3,4],[2,3,4,5]])
    frozen = bundle()
    for mode in ('prompt','reasoning'):
        x,y,ambient,background = control.collect_control(model,Tokenizer(),model.layer,rows(),c4,0,
            bundle=frozen,mode=mode,task_tokens=8)
        assert x.shape==(2,4) and y.shape==(2,3)
        assert ambient.shape==(16,4) and background.shape==(8,4)
        torch.testing.assert_close(ambient[:8],background)
        torch.testing.assert_close(ambient[8:10],torch.from_numpy(x))
        if mode=='reasoning':
            captured = control.capture_control(model,model.layer,
                pack([r['input_ids'] for r in rows()],Tokenizer(),'cpu'),rows(),mode)
            torch.testing.assert_close(ambient[10:12],torch.from_numpy(np.stack([r['terminal'] for r in captured])))


def test_source_order_tokens_splits_and_endings_cannot_drift():
    frozen = bundle()
    control.validate_bundle(frozen,rows())
    with pytest.raises(ValueError,match='parents/order'):
        control.validate_bundle(frozen,list(reversed(rows())))
    for field,value in [('source_sha256','other'),('split','validation'),('input_ids_sha256','bad')]:
        changed = rows();changed[0][field] = value
        with pytest.raises(ValueError,match='provenance'):
            control.validate_bundle({**frozen,'rows':changed},rows())


def test_legacy_adapter_preserves_collector_and_hashes_control_before_stages(tmp_path,monkeypatch):
    from sparsifier.eval import gsm8k_sequential as runner
    from sparsifier.eval.gsm8k import MODEL,REVISION
    frozen = {**bundle(),'model':MODEL,'revision':REVISION}
    frozen['manifest_sha256'] = digest(frozen)
    path = tmp_path/'bundle.json';write_immutable(path,frozen)
    calls = []
    def original_collect(*args,**kwargs):
        calls.append((args,kwargs))
        return 'unchanged'
    monkeypatch.setattr(runner,'collect',original_collect)
    original_write = runner.write_immutable
    def main():
        config = {'purpose':'development_sequential_MLP_retention','manifest_sha256':'parents'}
        runner.write_immutable(tmp_path/'config.json',config)
        assert config['calibration_control']['mode']=='legacy'
        assert config['calibration_control']['task_tokens']==4096
        assert digest(config)==digest(json.loads((tmp_path/'config.json').read_text()))
        assert runner.collect('model','tokenizer','layer',rows(),'c4',0,progress=None)=='unchanged'
    monkeypatch.setattr(runner,'main',main)
    monkeypatch.setattr(sys,'argv',['control','--reasoning-bundle',str(path),'--mode','legacy','--task-tokens','4096','--','--arm','local'])
    control.main()
    assert len(calls)==1 and calls[0][0][4]=='c4'
    assert runner.collect is original_collect and runner.write_immutable is original_write
