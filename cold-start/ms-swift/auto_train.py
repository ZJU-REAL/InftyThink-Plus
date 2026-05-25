import os
import shutil
import time 


def build_hf_to_mcore_cmd(config):
    model_path = config['load']
    mcore_path = f"{config['save']}/init-mcore"
    config['load'] = mcore_path

    if os.path.exists(mcore_path):
        shutil.rmtree(mcore_path)

    cmd = f"CUDA_VISIBLE_DEVICES=0 swift export --model {model_path} \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir {mcore_path} \
    --test_convert_precision true"

    if 'model_type' in config:
        cmd += f" --model_type {config['model_type']}"
    return cmd


def build_mcore_to_hf_cmd(config):
    cmd = f"CUDA_VISIBLE_DEVICES=0 swift export \
    --mcore_model {config['save']} \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir {os.path.join(config['save'], 'huggingface_format')} \
    --test_convert_precision true"
    return cmd


def build_main_cmd(config):
    cmd = f"export MODELSCOPE_CACHE={os.environ['MODELSCOPE_CACHE']} && \
        export MEGATRON_LM_PATH={os.environ['MEGATRON_LM_PATH']} && "
    cmd += "NPROC_PER_NODE=8 \
        NNODES=1 \
        NODE_RANK=0 \
        MASTER_ADDR=localhost \
        MASTER_PORT=32768 "
    cmd += "megatron sft "
    for k, v in config.items():
        cmd += '--{} {} '.format(k, v)
    return cmd


if __name__ == '__main__':
    import argparse 
    import json
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True)
    parser.add_argument('--model_name', '-n', default='example')
    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))
    config['save'] = os.path.join(config['save'], args.model_name)
    config['wandb_exp_name'] = args.model_name

    rank = 0
    print("Current RANK:", rank, flush=True)
    hf_to_mcore_cmd = build_hf_to_mcore_cmd(config)

    if rank == 0:
        print("Convert HF model to MCore model:", hf_to_mcore_cmd, flush=True)
        subprocess.check_call(hf_to_mcore_cmd, shell=True)
    else:
        while True:
            if os.path.exists(f"{config['load']}/latest_checkpointed_iteration.txt"):
                break
            time.sleep(30)
    
    main_cmd = build_main_cmd(config)
    print("Train Model:", main_cmd, flush=True)
    subprocess.check_call(main_cmd, shell=True)

    if rank == 0:
        mcore_to_hf_cmd =  build_mcore_to_hf_cmd(config)
        print("Convert MCore model to HF model:", mcore_to_hf_cmd, flush=True)
        subprocess.check_call(mcore_to_hf_cmd, shell=True)
