export output_dir="qwen2-7b-step-dpo"
export prompt="alpaca"

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml --mixed_precision bf16 \
    --num_processes 8 \
    train.py configs/config_full.yaml \
    --model_name_or_path="xinlai/Qwen2-7B-SFT" \
    --data_path="xinlai/Math-Step-DPO-10K" \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --torch_dtype=bfloat16 \
    --bf16=True \
    --beta=0.5 \
    --num_train_epochs=8 \
    --save_strategy='steps' \
    --save_steps=400 \
    --save_total_limit=1 \
    --output_dir=outputs/$output_dir \
    --hub_model_id=$output_dir \
    --prompt=$prompt

python eval_math.py --model outputs/$output_dir --data_file ./data/test/GSM8K_test_data.jsonl --save_path 'eval_results/gsm8k/'$output_dir'.json' --prompt $prompt --tensor_parallel_size 4

python eval_math.py --model outputs/$output_dir --data_file ./data/test/MATH_test_data.jsonl --save_path 'eval_results/math/'$output_dir'.json' --prompt $prompt --tensor_parallel_size 4
