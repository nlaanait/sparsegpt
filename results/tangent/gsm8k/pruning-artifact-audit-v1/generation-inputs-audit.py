"""Independently bind saved generation inputs to the frozen gold questions."""
from pathlib import Path
from hashlib import sha256
from copy import deepcopy
import json
from transformers import AutoTokenizer
from sparsifier.data.gsm8k import digest,load_manifest
from sparsifier.eval.gsm8k import MODEL,REVISION,render_prompt,atomic_json


def check_raw(raw,config_hash,ids,packed):
    assert raw['config_sha256']==config_hash
    assert raw['batch_ids']==ids
    assert raw['input_ids_sha256']==digest(packed)


def check_row(row,prompt,tokens,config):
    assert row['config_sha256']==digest(config)
    assert row['prompt_sha256']==digest(prompt)
    assert row['input_tokens']==len(tokens)
    assert row['batch_size']==len(row['batch_ids'])
    assert row['batching']==config['batching']


def main():
    root=Path.cwd();base=root/'results/tangent/gsm8k';eos=base/'eos-corrected-v1'
    frozen=json.loads((base/'pruning-artifact-audit-v1/frozen-source-v1/manifest.json').read_text())
    renderer=root/'src/sparsifier/eval/gsm8k.py'
    assert sha256(renderer.read_bytes()).hexdigest()==frozen['files']['src/sparsifier/eval/gsm8k.py']
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    assert tokenizer.pad_token_id==248044 and tokenizer.eos_token_id==248046
    journal=json.loads((eos/'submissions.json').read_text());outputs=[];negative_controls=0
    for task in journal['tasks']+journal['updates']:
        folder=Path(task['output']);path=folder/'answers.jsonl'
        if not path.exists():continue
        body=path.read_bytes();lines=body.decode().splitlines()
        if not body.endswith(b'\n'):lines=lines[:-1]
        rows=[json.loads(s) for s in lines]
        if not rows:continue
        config=json.loads((folder/'config.json').read_text());assert config['model']==MODEL and config['revision']==REVISION
        manifest=load_manifest(base/'data'/('decision64-light128-repcal-seed0-v1.json' if task['source'] is None else 'decision128-full512-repcal-seed0-batch8-v1.json'))
        assert config['manifest_sha256']==manifest['manifest_sha256']
        gold={r['id']:r for r in manifest['rows']};assert config['evaluation_ids']==[r['id'] for r in manifest['rows']]
        assert [r['id'] for r in rows]==config['evaluation_ids'][:len(rows)]
        batches={};raw_files={};size=config['batching']['batch_size']
        for i,row in enumerate(rows):
            start=(i//size)*size;ids=config['evaluation_ids'][start:start+size]
            assert row['batch_ids']==ids
            key=digest(ids)
            if key not in batches:
                prompts=[render_prompt(tokenizer,gold[identifier]['question']) for identifier in ids]
                sequences=[tokenizer.encode(p,add_special_tokens=False) for p in prompts]
                width=max(map(len,sequences));packed=[[tokenizer.pad_token_id]*(width-len(s))+s for s in sequences]
                raw_path=folder/'raw-generations'/(key+'.json');raw=json.loads(raw_path.read_text())
                check_raw(raw,digest(config),ids,packed)
                batches[key]=(prompts,sequences);raw_files[raw_path.name]=sha256(raw_path.read_bytes()).hexdigest()
                if negative_controls==0:
                    corrupted=dict(raw,input_ids_sha256='corrupted')
                    try:check_raw(corrupted,digest(config),ids,packed)
                    except AssertionError:negative_controls+=1
                    else:raise AssertionError('packed-input corruption was not detected')
            prompts,sequences=batches[key];position=ids.index(row['id'])
            check_row(row,prompts[position],sequences[position],config)
            if negative_controls==1:
                corrupted=deepcopy(row);corrupted['prompt_sha256']='corrupted'
                try:check_row(corrupted,prompts[position],sequences[position],config)
                except AssertionError:negative_controls+=1
                else:raise AssertionError('prompt corruption was not detected')
        result={'scope':'generation_input_provenance_only','evaluation':str(folder),'count':len(rows),
            'common128_complete':len(rows)>=128,'evaluation_report_present':(folder/'report.json').exists(),
            'config_sha256':digest(config),'answers_snapshot_sha256':sha256(body).hexdigest(),
            'audited_records_sha256':digest(rows),'manifest_sha256':manifest['manifest_sha256'],
            'verified_batches':len(batches),'packed_input_and_prompt_mismatches':0,
            'raw_files_sha256':raw_files,'renderer_sha256':sha256(renderer.read_bytes()).hexdigest(),
            'auditor_sha256':sha256(Path(__file__).read_bytes()).hexdigest()}
        output=eos/f"audits/{task['label']}-generation-inputs.json";atomic_json(output,result)
        outputs.append({'label':task['label'],'count':len(rows),'batches':len(batches),'audit':str(output)})
    assert negative_controls==2
    summary={'scope':'independent_gold_question_to_packed_input_check_not_goal_completion',
        'verified_evaluations':len(outputs),'evaluations':outputs,'in_memory_corruption_controls_detected':negative_controls,
        'uses_GPU':False,'auditor_sha256':sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_json(eos/'audits/generation-inputs-summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
