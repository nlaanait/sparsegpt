import pytest
from sparsifier.data.gsm8k_validation import reserve


def test_validation_excludes_sources_and_duplicate_questions():
    data = [{'question':f'question {i}', 'answer':'#### 3'} for i in range(1010)]
    data[1000]['question'] = ' QUESTION   2 '
    a = reserve(data,{2,1001},8,17)
    assert {r['source_index'] for r in a} == set(range(1002,1010))
    assert all(r['split']=='validation' for r in a)
    assert a == reserve(data,{2,1001},8,17)
    with pytest.raises(ValueError,match='insufficient'):
        reserve(data,{2,1001},9,17)
