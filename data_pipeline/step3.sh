export MODEL_PATH='/dataset/pretrained-models/Qwen2-7B-Instruct'
export EVAL_PROMPT='qwen2-boxed-prefix'
export JSON_FILE='./data_pipeline/continue_from_incorrect_step.jsonl'
export PRED_PATH='./data_pipeline/corrections/qwen2-7b-instruct-correction'
export SAVE_PATH='./data_pipeline/data.json'

python3 data_pipeline/prepare_for_correction.py --prompt $EVAL_PROMPT \
    --save_file $JSON_FILE \
    --generated_files "./data_pipeline/generated/*.json"

CUDA_VISIBLE_DEVICES=0 python eval_math.py --model $MODEL_PATH --remainder 0 --n_groups 8 --save_path $PRED_PATH"0.json" --data_file $JSON_FILE --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 20 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=1 python eval_math.py --model $MODEL_PATH --remainder 1 --n_groups 8 --save_path $PRED_PATH"1.json" --data_file $JSON_FILE --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 20 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=2 python eval_math.py --model $MODEL_PATH --remainder 2 --n_groups 8 --save_path $PRED_PATH"2.json" --data_file $JSON_FILE --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 20 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=3 python eval_math.py --model $MODEL_PATH --remainder 3 --n_groups 8 --save_path $PRED_PATH"3.json" --data_file $JSON_FILE --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 20 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=4 python eval_math.py --model $MODEL_PATH --remainder 4 --n_groups 8 --save_path $PRED_PATH"4.json" --data_file $JSON_FILE --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 20 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=5 python eval_math.py --model $MODEL_PATH --remainder 5 --n_groups 8 --save_path $PRED_PATH"5.json" --data_file $JSON_FILE --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 20 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=6 python eval_math.py --model $MODEL_PATH --remainder 6 --n_groups 8 --save_path $PRED_PATH"6.json" --data_file $JSON_FILE --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 20 --seed 0 --tensor_parallel_size 1 &
CUDA_VISIBLE_DEVICES=7 python eval_math.py --model $MODEL_PATH --remainder 7 --n_groups 8 --save_path $PRED_PATH"7.json" --data_file $JSON_FILE --prompt $EVAL_PROMPT --temp 0.8 --top_p 0.95 --rep 20 --seed 0 --tensor_parallel_size 1
