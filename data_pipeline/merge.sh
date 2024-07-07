export MODEL_PATH='/dataset/pretrained-models/Qwen2-7B-Instruct'
export EVAL_PROMPT='qwen2-boxed-prefix'
export JSON_FILE='./data_pipeline/continue_from_incorrect_step.jsonl'
export PRED_PATH='./data_pipeline/corrections/qwen2-7b-instruct-correction'
export SAVE_PATH='./data_pipeline/data.json'

python3 data_pipeline/generate_dataset.py --prompt $EVAL_PROMPT \
    --save_path $SAVE_PATH \
    --json_file $JSON_FILE \
    --corrected_files $PRED_PATH"*.json"
