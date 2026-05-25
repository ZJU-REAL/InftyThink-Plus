import datasets
import jsonlines

dataset = datasets.load_dataset('open-thoughts/OpenThoughts-114k', 'metadata')['train']


with jsonlines.open('OpenThoughts-114k.jsonl', 'w') as writer:
    for row in dataset:
        writer.write({
            "messages": [
                {"role": "user", "content": row["problem"]},
                {"role": "assistant", "content": f"<think>\n{row['deepseek_reasoning']}\n</think>\n{row['deepseek_solution']}"}
            ],
            "domain": row['domain'],
            "source": row['source'],
        })
