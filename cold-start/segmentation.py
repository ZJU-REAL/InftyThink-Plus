# -*- coding: utf-8 -*-
import re
import jsonlines
from typing import List


def should_segment(sentence):
    if sentence.lstrip()[0].isupper():
        return True
    return False


def split_thoughts(text: str) -> List[str]:
    """Reasoning Process Segmentation

    Args:
        text (str): the entire solution text

    Returns:
        List[str]: segmented reasoning process
    """
    steps = text.split('\n\n')
    steps = [step for step in steps if step.strip() != '']
    all_steps = [steps[0]]
    for step in steps[1:]:
        if should_segment(step):
            all_steps.append(step)
        else:
            all_steps[-1] = all_steps[-1] + '\n\n' + step
    return all_steps


def process(inst, eta=4096):
    try:
        assert len(inst['messages']) == 2
        question = inst['messages'][0]['content']
        solution = inst['messages'][1]['content']

        thoughts, conclusion = re.search(
            r'^<think>\n(.+)\n</think>(.+)$', solution, re.S
        ).groups()

        thoughts = split_thoughts(thoughts)  # list

        tokens = tokenizer(thoughts).input_ids
        lengths = [len(t) for t in tokens]
        assert all([l < eta+500 for l in lengths]), str(lengths)

        idx_span = []
        start = 0
        end = 0
        while end < len(lengths):
            end += 1
            if sum(lengths[start:end]) > eta:
                idx_span.append((start, end))
                start = end
        else:
            if end > start:
                idx_span.append((start, end))
        return {
            "question": question,
            "solution": solution,
            "thoughts": thoughts,  # list
            "conclusion": conclusion,
            "thoughts_span": ['\n\n'.join(thoughts[s:e]) for s, e in idx_span],
            "span_number": len(idx_span),
            "span_idx": idx_span,
            "total_thoughts": len(lengths),
            "span_length": [sum(lengths[s:e]) for s, e in idx_span],
        }
    except Exception as e:
        print(e)
        # print(inst)
        # we found some bad case in OpenR1-Math,
        # ignore them, just a few thousand samples
        return False


if __name__ == '__main__':
    import argparse
    import transformers
    from datasets import load_dataset
    from functools import partial

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-d', type=str, required=True)
    parser.add_argument('--tokenizer', '-t', type=str,
                        default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    parser.add_argument('--eta', type=int, choices=[2048, 4096, 6144, 8192],
                        default=6144)
    parser.add_argument('--output_path', '-o', type=str,)
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    # Load dataset with hf datasets
    dataset = load_dataset('json', data_files=args.dataset_path, num_proc=64)
    print(f"Before process: {len(dataset['train'])}")

    # process
    infty_dataset = dataset.filter(
        partial(process, eta=args.eta), num_proc=128,)
    infty_dataset = infty_dataset.map(
        partial(process, eta=args.eta), num_proc=128,)
    print(f"After process: {len(infty_dataset['train'])}")

    # save to disk
    with jsonlines.open(args.output_path, 'w') as writer:
        for row in infty_dataset['train']:
            writer.write(row)
