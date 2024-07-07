export SAVE_PATH='/dataset/pretrained-models/Qwen2-7B-Instruct'
export MATH_EVAL_SAVE_PATH='./data_pipeline/predictions/qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group'
export EVAL_PROMPT='qwen2-boxed-step'

# python eval_gsm8k.py --model $SAVE_PATH --data_file ./data/test/GSM8K_test.jsonl
CUDA_VISIBLE_DEVICES=0 python eval_math.py --model $SAVE_PATH --remainder 0 --n_groups 8 --save_path $MATH_EVAL_SAVE_PATH"0.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=1 python eval_math.py --model $SAVE_PATH --remainder 1 --n_groups 8 --save_path $MATH_EVAL_SAVE_PATH"1.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=2 python eval_math.py --model $SAVE_PATH --remainder 2 --n_groups 8 --save_path $MATH_EVAL_SAVE_PATH"2.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=3 python eval_math.py --model $SAVE_PATH --remainder 3 --n_groups 8 --save_path $MATH_EVAL_SAVE_PATH"3.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=4 python eval_math.py --model $SAVE_PATH --remainder 4 --n_groups 8 --save_path $MATH_EVAL_SAVE_PATH"4.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=5 python eval_math.py --model $SAVE_PATH --remainder 5 --n_groups 8 --save_path $MATH_EVAL_SAVE_PATH"5.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=6 python eval_math.py --model $SAVE_PATH --remainder 6 --n_groups 8 --save_path $MATH_EVAL_SAVE_PATH"6.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=7 python eval_math.py --model $SAVE_PATH --remainder 7 --n_groups 8 --save_path $MATH_EVAL_SAVE_PATH"7.json" --data_file /dataset/industry_gpt/llm_infer/AQuA/train_qa.jsonl --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 2 --seed 0 --tensor_parallel_size 1
