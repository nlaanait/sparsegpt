"""Reserve original training problems for sparsity selection; never reads test."""
from __future__ import annotations
import argparse
import hashlib
import random
from pathlib import Path
from sparsifier.data.gsm8k import digest, load_manifest, parse_answer, write_immutable


def reserve(dataset, excluded_indices, count, seed):
    # Exclude exact normalized question duplicates across the boundary as well.
    normalize = lambda s: ' '.join(s.casefold().split())
    seen = {normalize(dataset[i]['question']) for i in excluded_indices}
    eligible = [i for i in range(1000,len(dataset)) if i not in excluded_indices]
    random.Random(seed).shuffle(eligible)
    rows = []
    for i in eligible:
        item = dataset[i]; question = normalize(item['question'])
        if question in seen: continue
        answer = parse_answer(item['answer'])
        if answer is None: raise ValueError(f'invalid answer at {i}')
        seen.add(question)
        rows.append({'id':f'gsm8k/train/{i}','source_index':i,'family':f'gsm8k/train/{i}',
                     'split':'validation','question':item['question'],'answer':str(answer),
                     'source_sha256':digest(item)})
        if len(rows) == count: return rows
    raise ValueError('insufficient independent source questions')


def main():
    from datasets import Dataset
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-arrow',type=Path,required=True)
    p.add_argument('--exclude-manifest',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=512)
    p.add_argument('--seed',type=int,default=1701)
    args = p.parse_args()
    if args.train_arrow.name != 'gsm8k-train.arrow' or args.count < 1:
        p.error('positive count and GSM8K train source required')
    excluded, manifests = set(), []
    for path in args.exclude_manifest:
        m = load_manifest(path)
        if m['source_split'] != 'train': raise ValueError('only training manifests supported')
        excluded.update(r['source_index'] for r in m['rows'])
        manifests.append(m['manifest_sha256'])
    dataset = Dataset.from_file(str(args.train_arrow))
    rows = reserve(dataset,excluded,args.count,args.seed)
    result = {'schema':1,'dataset':'openai/gsm8k','source_split':'train',
              'purpose':'original_GSM8K_validation_for_sparsity_selection_not_final_test',
              'source_file_sha256':hashlib.sha256(args.train_arrow.read_bytes()).hexdigest(),
              'excluded_manifests':manifests,'excluded_indices':sorted(excluded),
              'minimum_source_index':1000,'seed':args.seed,'rows':rows,
              'deduplication':'exact casefolded whitespace-normalized question; semantic duplicates not guaranteed absent'}
    result['manifest_sha256'] = digest(result)
    write_immutable(args.output,result)
    print(f'Reserved {len(rows)} original training problems')


if __name__ == '__main__': main()
