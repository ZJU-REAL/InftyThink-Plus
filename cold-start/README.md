# Recipe for cold-start stage

This recipe provides step-by-step guidance for the cold-start stage of InftyThink+.

## Step 1: Paradigm Transformation
### Step 1.1: Reasoning Process Partition
For each data instance, we partition the original reasoning process ($r$) into a sequence of shorter segments, guided by a hyperparameter η that specifies the maximum token length allowed per segment.
Instead of performing naive or arbitrary truncation, we adopt a semantically-aware segmentation strategy. Specifically, we first decompose the reasoning process into fine-grained semantic units by detecting natural boundaries such as sentence or paragraph breaks.

```bash
python3 to_jsonl.py # save dataset in jsonl format
python3 segmentation.py -d ./OpenThoughts-114k.jsonl -o ./OpenThoughts-114k-seg-6k.jsonl # partition
```


### Step 1.2: Summary Generation
For each reasoning segment, we construct a concise summary that distills its key insights and reflects the incremental progress toward the final solution. We adopt a high-capacity foundation model M for summary generation, specifically, Qwen3-4B-Instruct-2507. All summaries are generated using carefully designed prompts.

```bash
python3 inftythink_runner.py -m Qwen/Qwen3-4B-Instruct-2507 -i ./OpenThoughts-114k-seg-6k.jsonl -o ./OpenThoughts-114k-seg-6k-out.jsonl --max_compression_tokens 1024
```

### Step 1.3: Training Instance Construction
Based on the segmented reasoning traces and their corresponding summaries, we construct a set of training instances that explicitly supervise the model to perform iterative reasoning with intermediate summarization.

```bash
python3 format_inftythink.py -i ./OpenThoughts-114k-seg-6k-out.jsonl -o ./OpenThoughts-114k-seg-6k-inftythink-train.jsonl
```

## Step 2: Supervised Fine-tuning
We use ms-swift as our SFT training backend.

```bash
cd ms-swift
# please refer to https://swift.readthedocs.io/zh-cn/latest/ to setup the training environment
export MODELSCOPE_CACHE=xxx
export MEGATRON_LM_PATH=xxx
python3 auto_train.py -c examples/config/DeepSeek-R1-Distill-Qwen-1.5B-OpenThoughts-114k-vanilla-bs8.json # for vanilla
python3 auto_train.py -c examples/config/DeepSeek-R1-Distill-Qwen-1.5B-OpenThoughts-114k-inftythink-6k-bs8.json # for InftyThink
```