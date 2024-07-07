export OPENAI_BASE_URL="" # input openai base_url here
export OPENAI_API_KEY="" # input openai api_key here

python3 data_pipeline/locate_error_by_gpt4.py \
    --prompt "qwen2-boxed-step" \
    --save_dir "./data_pipeline/generated" \
    --json_files "./data_pipeline/predictions/qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group*.json" \
    --max_count_total 100
