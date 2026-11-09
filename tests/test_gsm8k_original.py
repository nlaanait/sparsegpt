import hashlib
import json
import pytest
import torch
from sparsifier.data.gsm8k import digest
from sparsifier.eval.gsm8k import MODEL, REVISION
from sparsifier.eval.gsm8k_original import frozen_stages


def test_frozen_chain_detects_modified_weights(tmp_path):
    config = {'model':MODEL,'revision':REVISION}
    (tmp_path/'config.json').write_text(json.dumps(config))
    candidate = tmp_path/'site-000.pt'
    torch.save({'config_sha256':digest(config),'upstream':digest(config),
                'site':[0,'mlp.down_proj'],'weight':torch.ones(2,3)},candidate)
    report = {'complete':True,'config_sha256':digest(config),'stages':[
        {'site':[0,'mlp.down_proj'],'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest()}]}
    (tmp_path/'report.json').write_text(json.dumps(report))
    assert frozen_stages(tmp_path) == (config,report)
    torch.save({'weight':torch.zeros(2,3)},candidate)
    with pytest.raises(ValueError,match='content hash'):
        frozen_stages(tmp_path)
