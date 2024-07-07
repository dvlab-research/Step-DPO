import argparse
import json
import glob
import jsonlines

def main(args):
    save_path = args.save_path
    json_files = sorted(glob.glob(args.corrected_files))
    identifier2items = {}
    for json_file in json_files:
        with open(json_file) as f:
            for item in json.load(f):
                if item['result']:
                    if 'alpaca' in args.prompt:
                        prompt = item['prompt'].split("### Instruction:")[1].split("### Response:")[0].strip()
                        prefix = item['prompt'].split("### Response:")[-1].lstrip()
                    elif 'qwen2-boxed' in args.prompt:
                        prompt = item['prompt'].split("<|im_start|>user\n")[1].split("\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>")[0].strip()
                        prefix = item['prompt'].split("<|im_start|>assistant\n")[-1].lstrip()
                    else:
                        raise NotImplementedError("Prompt {} is not supported currently".format(args.prompt))

                    prefix = prefix.replace("Let's think step by step.\n", "")
                    identifier = prompt + "||" + prefix
                    if identifier not in identifier2items:
                        identifier2items[identifier] = []
                    identifier2items[identifier].append(item)

    new_items = []
    invalid_cnt = 0
    cnt = 0
    with jsonlines.open(args.json_file, "r") as f:
        for line in f:
            prompt = line['instruction']
            prefix = line['prefix']
            identifier = prompt + "||" + prefix

            if identifier not in identifier2items:
                invalid_cnt += 1
                continue
            items = identifier2items[identifier]
            visited_chosen = set()
            for item in items:
                cnt += 1
                chosen = item['completion']
                rejected = line['output']

                chosen_first_step = chosen.split("\nStep ")[0]
                rejected_first_step = rejected.split("\nStep ")[0]

                if chosen_first_step in visited_chosen:
                    continue
                
                visited_chosen.add(chosen_first_step)

                new_item = {
                    'dataset': line['type'],
                    'prompt': prompt,
                    'prefix': "Let's think step by step.\n" + prefix,
                    'chosen': chosen_first_step,
                    'rejected': rejected_first_step,
                    'original_chosen': chosen,
                    'answer': line['answer'],
                }
                new_items.append(new_item)

    print("len(new_items): {}, invalid_cnt: {}, cnt: {}".format(len(new_items), invalid_cnt, cnt))
    with open(save_path, "w+") as f:
        json.dump(new_items, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default='qwen2-boxed-step')
    parser.add_argument("--save_path", type=str, default='./data_pipeline/data.json')
    parser.add_argument("--json_file", type=str, default="./data_pipeline/continue_from_incorrect_step.jsonl")
    parser.add_argument("--corrected_files", type=str, default="./data_pipeline/corrections/qwen2-7b-instruct-correction*.json")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
