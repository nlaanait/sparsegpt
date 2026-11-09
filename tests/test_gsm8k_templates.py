from fractions import Fraction
import itertools
import pytest

from sparsifier.data.gsm8k_templates import TEMPLATES, arithmetic


def test_no_code_execution_and_exact_arithmetic():
    assert arithmetic("a/3+1/6", {"a": 1}) == Fraction(1, 2)
    with pytest.raises(ValueError):
        arithmetic("__import__('os').system('true')", {})


def test_all_reviewed_templates_have_valid_crossed_grids():
    for template in TEMPLATES:
        template.render(template.parameters())
        accepted = []
        for values in itertools.product(*template.ranges):
            parameters = dict(zip(template.parameters(), values))
            try:
                question, answer = template.render(parameters)
            except ValueError:
                continue
            assert "{" not in question
            assert Fraction(answer) >= 0
            accepted.append(parameters)
        assert len(accepted) >= 8, template.source


def test_related_templates_share_leakage_group():
    by_source = {t.source: t for t in TEMPLATES}
    assert by_source[208].group == by_source[209].group
    assert by_source[203].group == by_source[219].group == by_source[222].group
    assert by_source[207].group == by_source[217].group
