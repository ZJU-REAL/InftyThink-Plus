import asyncio
import argparse
import jsonlines
import os
import sglang as sgl
import torch
import transformers
import uuid
from tqdm import tqdm


HISTORY_START = '<history>'
HISTORY_END = '</history>'

THINK_START = '<think>'
THINK_END = '</think>'

SUMMARY_START = '<summary>'
SUMMARY_END = '</summary>'

CONCLUSION_START = ''
CONCLUSION_END = ''


with open('prompt/step1.txt', 'r') as f:
    PROMPT_STEP_1 = f.read()

with open('prompt/step2.txt', 'r') as f:
    PROMPT_STEP_2 = f.read()


async def generate(predictor, tokenizer, sampling_params, row):
    all_cases_for_inference = []
    # prepare data for generation
    if not isinstance(row['thoughts_span'], list):
        raise ValueError
    _id = row.get('uuid', str(uuid.uuid4()))
    for span_idx, span in enumerate(row['thoughts_span']):
        all_cases_for_inference.append({
            "question": row['question'].strip(),
            # "raw_solution": row['answer'].strip(),
            "conclusion": row['conclusion'].strip(),
            
            "reasoning": span.strip(),
            "uuid": _id,
        })
    
    all_data = []
    last_summary = None

    for index, inst in enumerate(all_cases_for_inference):
        if len(all_cases_for_inference) == 1:
            messages = None
        else:
            if index == 0:
                messages = [
                    {"role": "user", "content": inst['question']},
                    {"role": "assistant", "content": inst['reasoning']},
                    {"role": "user", "content": PROMPT_STEP_1}
                ]
            elif index < len(all_cases_for_inference) - 1:
                messages = [
                    {"role": "user", "content": inst['question']},
                    {"role": "assistant", "content": last_summary},
                    {"role": "user", "content": "Please continue your reasoning based on your past reasoning history."},
                    {"role": "assistant", "content": inst['reasoning']},
                    {"role": "user", "content": PROMPT_STEP_2}
                ]
            else:
                messages = None
            
        
        if messages is not None:
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            resp = await predictor.async_generate(prompt, sampling_params)
            finish_reason = resp['meta_info']['finish_reason']['type']
            summary = resp['text']
            
            for _ in range(64): # 重试
                if finish_reason == 'stop':
                    break
                resp = await predictor.async_generate(prompt, sampling_params)
                finish_reason = resp['meta_info']['finish_reason']['type']
                summary = resp['text']
            else:
                return []
            
            all_data.append({
                **inst,
                "uuid": f"{inst['uuid']}-InftyThink-{index}-of-{len(all_cases_for_inference)}", 
                "history": last_summary,
                "reasoning": inst['reasoning'],
                "summary": summary,
                "conclusion": None
            })
            last_summary = summary
        else:
            all_data.append({
                **inst,
                "uuid": f"{inst['uuid']}-InftyThink-{index}-of-{len(all_cases_for_inference)}", 
                "history": last_summary,
                "reasoning": inst['reasoning'],
                "summary": None,
                "conclusion": inst['conclusion']
            })
    return all_data


def filter_uuid(data, existing_ids):
    if data['uuid'] in existing_ids:
        return False
    return True


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", '-m', type=str,)
    parser.add_argument("--data_path", '-i', type=str,)
    parser.add_argument("--output_path", '-o', type=str,)
    parser.add_argument("--tp_size", '-tp', type=int, default=1)
    parser.add_argument("--max_compression_tokens", type=int, default=32768)# 32768 for no limit
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    predictor = sgl.Engine(
        model_path=args.model, 
        dp_size=torch.cuda.device_count() // args.tp_size, tp_size=args.tp_size, 
        mem_fraction_static=0.9
    )
    sampling_params = {
        "temperature": 0.5,
        "max_new_tokens": args.max_compression_tokens,
        "top_p": 0.95
    }
    print(predictor)

    with jsonlines.open(args.data_path, 'r') as reader:
        dataset = list(reader)
    total = len(dataset)
    print(f"Before process: {total}")

    if os.path.exists(args.output_path):
        existing_ids = set()
        with jsonlines.open(args.output_path, 'r') as f:
            for line in f:
                existing_ids.add(line['uuid'].split('-InftyThink')[0])
        dataset = [data for data in dataset if filter_uuid(data, existing_ids)]
    remain = len(dataset)
    print(f"Remove completed cases: {remain}")

    sem = asyncio.Semaphore(4096) 
    async def wrapper(*args):
        async with sem:
            return await generate(*args)

    tasks = [wrapper(predictor, tokenizer, sampling_params, row) for row in dataset]
    with jsonlines.open(args.output_path, "a") as writer:
        for res in tqdm(asyncio.as_completed(tasks), total=total, desc="Generate", initial=(total-remain)):
            writer.write_all(await res)

if __name__ == '__main__':
    asyncio.run(main())
