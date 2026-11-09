"""Reviewed arithmetic programs adapted from GSM8K training sources.

This first registry is a development-scale controlled study. Family counts
must be expanded/powered before making confirmatory generalization claims.
Related algebraic templates share a group and cannot cross dataset splits.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from fractions import Fraction
import itertools
import random
from pathlib import Path

from sparsifier.data.gsm8k import digest, parse_answer, write_immutable


def arithmetic(expression: str, parameters: dict[str, int]) -> Fraction:
    """Evaluate only rational arithmetic, never Python eval or source code."""
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Fraction(node.value)
        if isinstance(node, ast.Name):
            return Fraction(parameters[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
        raise ValueError("unsupported arithmetic syntax")
    return visit(ast.parse(expression, mode="eval"))


@dataclass(frozen=True)
class Template:
    source: int
    group: str
    text: str
    expression: str
    defaults: tuple[int, ...]
    ranges: tuple[tuple[int, ...], ...]
    constraints: tuple[str, ...] = ()  # expressions must be nonnegative
    integer_answer: bool = True

    def parameters(self):
        return dict(zip("abcdef", self.defaults))

    def render(self, parameters):
        if any(arithmetic(c, parameters) < 0 for c in self.constraints):
            raise ValueError("invalid physical quantities")
        answer = arithmetic(self.expression, parameters)
        if answer < 0 or (self.integer_answer and answer.denominator != 1):
            raise ValueError("answer outside template domain")
        return self.text.format(**parameters), str(answer)


# Defaults reproduce the source's numeric answer. Wording removes ambiguities
# and uses explicit units; expressions were checked against the source solution.
TEMPLATES = (
    Template(200, "weighted_sales", "An artist sells small portraits for ${a} each and large portraits for twice that price. Each day she sells {b} small and {c} large portraits. What is her revenue in dollars over {d} days?", "(a*b+2*a*c)*d", (5,3,5,3), ((4,5,6,7),(2,3,4,5),(3,4,5,6),(2,3,4,5))),
    Template(202, "percent_increase", "A savings account contains ${a}. Its owner adds an amount equal to {b}% of that balance. What is the new balance in dollars?", "a*(100+b)/100", (200,20), ((100,200,300,400),(10,20,30,40))),
    Template(203, "remaining_per_item", "A customer buys {b} drinks for ${c} each and {d} candy bars, spending ${a} in total. What is the price of each candy bar in dollars?", "(a-b*c)/d", (28,2,4,5), ((28,40,60,80),(1,2,3,4),(2,3,4,5),(2,4,5,10)), ("a-b*c",)),
    Template(204, "product_division", "A stack of {c} books is {a} inches thick. Every inch contains {b} pages. How many pages does each book contain on average?", "a*b/c", (12,80,6), ((8,10,12,16),(40,60,80,100),(2,4,5,8))),
    Template(205, "half_plus_product", "A host invites {a} guests. All attend, and half bring one additional guest each. Each person uses a new plate for each of {b} courses. How many plates are needed?", "(a+a/2)*b", (30,3), ((20,24,30,40),(2,3,4,5))),
    Template(206, "reverse_population", "A reserve has {a} lions after 12 months. Exactly {b} cubs were born and {c} lions died each month. How many lions were there initially?", "a-(b-c)*12", (148,5,1), ((120,148,180,220),(3,4,5,6),(1,2,3,4)), ("b-c","a-(b-c)*12")),
    Template(207, "pure_product", "A volunteer works {a} times each month for {b} hours each time. How many hours does the volunteer work in 12 months?", "a*b*12", (2,3), ((1,2,3,4),(2,3,4,5))),
    Template(208, "product_minus", "A saver deposits ${a} each month for {b} years. The saver then spends ${c}. How many dollars remain?", "a*b*12-c", (25,2,400), ((20,25,30,40),(2,3,4,5),(100,200,300,400)), ("a*b*12-c",)),
    Template(209, "product_minus", "There are {a} vans, each carrying {b} students. Of all the students, {c} are boys and the rest are girls. How many girls are there?", "a*b-c", (5,28,60), ((4,5,6,7),(20,24,28,32),(20,40,60,80)), ("a*b-c",)),
    Template(212, "practice_rate", "A musician needs {a} total hours of practice between ages {c} and {b}. Every year has 52 weeks, with {d} weeks off. The musician practices 5 days per working week. How many hours per practice day are needed?", "a/((b-c)*(52-d)*5)", (10000,20,12,2), ((6000,8000,10000,12000),(20,22,24,26),(10,12,14,16),(2,4,6,8)), ("b-c-1",), False),
    Template(213, "two_period_sum", "In the morning, one gardener plants {a} seeds and another plants {b} times as many. In the afternoon, the first plants {c} seeds and the second plants {d} fewer than the first. How many seeds do they plant altogether?", "a*(1+b)+2*c-d", (50,2,60,20), ((20,30,40,50),(2,3,4,5),(40,50,60,70),(10,20,30,40)), ("c-d",)),
    Template(214, "anniversary", "In {c} months, a couple will celebrate their {a}-year anniversary. How many months ago was their {b}-year anniversary?", "(a-b)*12-c", (4,2,6), ((4,5,6,7),(1,2,3,4),(2,4,6,8)), ("(a-b)*12-c",)),
    Template(215, "percent_difference", "A club has {a} members. Adults make up {b}% and the rest are children. How many more children than adults are there?", "a*(100-2*b)/100", (120,40), ((100,120,200,300),(10,20,30,40))),
    Template(216, "geometric_sum_inverse", "Over four years a saver deposits ${a} in total. Each year after the first, the amount deposited is {b} times the previous year's amount. How much was deposited in the first year?", "a/(1+b+b*b+b*b*b)", (450,2), ((150,300,450,600),(2,3,4,5)), (), False),
    Template(217, "pure_product", "Anne is {a} times as old as Emile. Emile is {b} times as old as Maude. Maude is {c} years old. How old is Anne?", "a*b*c", (2,6,8), ((2,3,4,5),(2,3,4,6),(2,4,6,8))),
    Template(219, "remaining_per_item", "An eating contest lasts {c} minutes. Halfway through, a contestant has eaten {b} items and must reach {a} items. How many items per minute must the contestant eat in the remaining time?", "(a-b)/(c/2)", (75,20,10), ((60,75,90,100),(10,20,30,40),(4,6,8,10)), ("a-b",), False),
    Template(220, "fuel_rate", "A car uses {b} gallons for every {c} miles. It must travel {a} miles each way on a round trip and already has {d} gallons. How many more gallons are needed?", "2*a*b/c-d", (600,20,400,8), ((400,600,800,1000),(10,20,30,40),(200,400,500,800),(2,4,6,8)), ("2*a*b/c-d",), False),
    Template(222, "remaining_per_item", "A student must read {a} pages and has already read {b}. There are {c} days remaining. How many pages per day must the student read?", "(a-b)/c", (408,113,5), ((200,300,408,500),(20,50,100,113),(2,4,5,10)), ("a-b",), False),
    Template(223, "equal_spending", "Two friends spend the same amount. The first buys a drink for ${a} and two burgers for ${b} each. The second buys sandwiches costing ${c} in total and one fruit drink. How many dollars does the fruit drink cost?", "a+2*b-c", (1,2,3), ((1,2,3,4),(2,3,4,5),(1,2,3,4)), ("a+2*b-c",)),
    Template(225, "weighted_sum", "A washer uses {a} gallons for a heavy wash, {b} for a regular wash, and {c} for a light wash. There are two heavy loads, three regular loads, and one light load. Two loads each need one additional light rinse. How many gallons are used?", "2*a+3*b+3*c", (20,10,2), ((20,24,28,32),(8,10,12,14),(2,3,4,5))),
    Template(226, "table_partition", "A hall contains {a} tables. Half have two chairs each, {b} tables have three chairs each, and the rest have four chairs each. How many chairs are in the hall?", "a+3*b+(a/2-b)*4", (32,5), ((20,24,28,32),(2,3,4,5)), ("a/2-b",)),
    Template(227, "debt_sum", "A winner receives ${a}. They pay ${b} to one friend, twice that amount to a second, and half the second payment to a third. How many dollars remain?", "a-4*b", (100,20), ((100,120,160,200),(10,15,20,25)), ("a-4*b",)),
    Template(229, "fraction_remaining", "A worker cuts {a} feet of cable into {b}-foot sections. The worker gives away 1/{c} of the sections and stores half the remaining sections. How many sections remain on hand?", "a/b*(1-1/c)/2", (1000,25,4), ((600,800,1000,1200),(10,20,25,50),(2,3,4,5))),
)


def build_controlled(source: Path, output: Path, seed=17, max_rows_per_family=128):
    from datasets import Dataset
    if source.name != "gsm8k-train.arrow":
        raise ValueError("controlled templates require training sources")
    source_data = Dataset.from_file(str(source))
    groups = sorted({t.group for t in TEMPLATES})
    random.Random(seed).shuffle(groups)
    # Development allocation, explicitly too small for final inference.
    names = ("geometry_fit", "pruning_calibration", "validation", "stage2_holdout", "controlled_test")
    splits = {g: names[i % len(names)] for i, g in enumerate(groups)}
    rows, pairs, registry = [], [], []
    for template in TEMPLATES:
        src = source_data[template.source]
        if str(parse_answer(src["answer"])) != template.render(template.parameters())[1]:
            raise ValueError(f"source program mismatch: {template.source}")
        registry.append({**asdict(template), "source_sha256": digest(src),
                         "split": splits[template.group], "review": "source_default_and_constraints_checked"})
        names_p = list(template.parameters())
        candidates = list(itertools.product(*template.ranges))
        random.Random(seed + template.source).shuffle(candidates)
        accepted = {}
        for values in candidates:
            parameters = dict(zip(names_p, values))
            try:
                question, answer = template.render(parameters)
            except (ValueError, ZeroDivisionError):
                continue
            identifier = f"controlled/{template.source}/{digest(parameters)[:16]}"
            row = {"id": identifier, "family": template.group, "source_index": template.source,
                   "split": splits[template.group], "parameters": parameters,
                   "question": question, "answer": answer, "expression": template.expression,
                   "source_sha256": digest(src)}
            accepted[values] = row
            if len(accepted) == max_rows_per_family:
                break
        rows.extend(accepted.values())
        for values, row in accepted.items():
            for k, factor in enumerate(names_p):
                axis = template.ranges[k]
                j = axis.index(values[k])
                if j + 1 == len(axis):
                    continue
                neighbor = list(values)
                neighbor[k] = axis[j + 1]
                other = accepted.get(tuple(neighbor))
                if other:
                    pairs.append({"left": row["id"], "right": other["id"],
                                  "factor": f"{template.group}/{factor}",
                                  "step": neighbor[k] - values[k], "split": row["split"]})
    manifest = {"schema": 1, "dataset": "gsm8k_training_controlled_v1",
                "source_split": "train", "seed": seed,
                "purpose": "development_scale_not_powered_confirmation",
                "templates": registry, "groups": splits, "rows": rows, "pairs": pairs}
    manifest["manifest_sha256"] = digest(manifest)
    write_immutable(output, manifest)
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-arrow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_controlled(args.train_arrow, args.output)
    print(f"{len(manifest['templates'])} templates, {len(manifest['groups'])} source groups, "
          f"{len(manifest['rows'])} instances, {len(manifest['pairs'])} paired changes")


if __name__ == "__main__":
    main()
