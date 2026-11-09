import pytest
from sparsifier.data.gsm8k import digest
from sparsifier.eval.gsm8k_reevaluate import validate_source


def test_reuses_exact_validation_ids_and_rejects_final_split():
    capture = {'block': 15}
    manifest = {'manifest_sha256': 'x', 'rows': [
        {'id': 'a', 'split': 'validation'}, {'id': 'b', 'split': 'controlled_test'}]}
    config = {'manifest_sha256': 'x', 'capture_sha256': digest(capture), 'evaluation_ids': ['a']}
    report = {'complete': True, 'config_sha256': digest(config)}
    assert validate_source(config, report, manifest, capture) == [manifest['rows'][0]]
    config['evaluation_ids'] = ['b']
    report['config_sha256'] = digest(config)
    with pytest.raises(ValueError, match='restricted to validation'):
        validate_source(config, report, manifest, capture)
    config['evaluation_ids'] = ['a']
    with pytest.raises(ValueError, match='mismatched'):
        validate_source(config, report, manifest, capture)
