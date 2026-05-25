# -*- coding: utf-8 -*-
from datasets import load_dataset, Features, Value

HISTORY_START = '<history>'
HISTORY_END = '</history>'

THINK_START = '<think>'
THINK_END = '</think>'

SUMMARY_START = '<summary>'
SUMMARY_END = '</summary>'

CONCLUSION_START = ''
CONCLUSION_END = ''

features = Features({
    "uuid": Value("string"),
    "question": Value("string"),
    "raw_solution": Value("string"),
    "history": Value("string"),
    "reasoning": Value("string"),
    "summary": Value("string"),
    "conclusion": Value("string"),
})


def filter_inftythink(inst):
    if inst['reasoning']:
        if inst['conclusion']:
            return not inst['summary']
        elif inst['summary']:
            return True
        else:
            return False
    else:
        return False


def format_inftythink(inst):
    messages = []
    messages.append({"role": "user", "content": inst['question']})

    resp = ""
    if inst['history']:
        resp += f"{HISTORY_START}\n{inst['history']}\n{HISTORY_END}"

    resp += f"{THINK_START}\n{inst['reasoning']}\n{THINK_END}"

    if inst['conclusion']:
        assert inst['summary'] is None, str(inst)
        resp += f"{CONCLUSION_START}\n{inst['conclusion']}\n{CONCLUSION_END}"
    else:
        assert inst['summary'] is not None, str(inst)
        resp += f"{SUMMARY_START}\n{inst['summary']}\n{SUMMARY_END}"

    messages.append({"role": "assistant", "content": resp})

    return {
        "messages": messages
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-i', type=str, required=True)
    parser.add_argument('--output_path', '-o', type=str, required=True)
    args = parser.parse_args()

    dataset = load_dataset('json', data_files=args.dataset_path,
                           features=features, num_proc=64)['train']
    print(f"Before process: {len(dataset)}")
    print(dataset)

    dataset = dataset.filter(filter_inftythink,  num_proc=64)
    print(dataset)

    dataset = dataset.map(format_inftythink, num_proc=64,
                          remove_columns=dataset.column_names)
    print(dataset)

    dataset = dataset.shuffle(seed=42)
    dataset.to_json(args.output_path, lines=True,
                    orient="records", force_ascii=False,)
