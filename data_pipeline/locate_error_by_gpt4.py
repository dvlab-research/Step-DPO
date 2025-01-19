import argparse
import glob
import json
import os
import time

import openai
import tqdm

client = openai.OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

prompt = '''### Problem:
{problem}

### Correct solution:
{solution}

### Incorrect answer:
{answer}

---

A math problem and its correct solution are listed above. We also give another incorrect answer, where step-by-step reasoning process is shown. Please output the correctness for each reasoning step in the given answer.

Requirements:
1. You should first output a step-by-step analysis process (no more than 200 words), and finally output the decision ("correct", "neutral", "incorrect") for each step following the format of "Final Decision:\nStep 1: correct; Step 2: neutral; ...";
2. Stop when you find the first incorrect step.'''

def main(args):

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    save_dir = args.save_dir
    visited_dirs = save_dir if len(args.visited_dirs) == 0 else args.visited_dirs
    json_files = sorted(glob.glob(args.json_files))

    pred_data = []
    for json_file in json_files:
        with open(json_file) as f:
            for item in json.load(f):
                if not item['result']:
                    pred_data.append(item)

    n_groups = args.n_groups
    remainder = args.remainder

    print("n_groups: {}, remainder: {}".format(n_groups, remainder))
    print("len(pred_data): ", len(pred_data))

    cnt = 0
    question2cnt = dict()
    for idx, pred_dict in tqdm.tqdm(enumerate(pred_data)):

        if 'alpaca' in args.prompt:
            question = pred_dict['prompt'].split("### Instruction:")[1].split("### Response:")[0].strip()
        elif 'qwen2-boxed' in args.prompt:
            question = pred_dict['prompt'].split("<|im_start|>user\n")[1].split("\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>")[0].strip()
        else:
            raise NotImplementedError("Prompt {} is not supported currently".format(args.prompt))

        if question in question2cnt and question2cnt[question] > args.max_count_per_question:
            continue
        if question not in question2cnt:
            question2cnt[question] = 0
        question2cnt[question] += 1

        # skip the invalid questions without diagram
        if "diagram" in question and 'asy' not in question:
            continue

        # skip other threads
        if idx % n_groups != remainder:
            continue

        # skip the visited questions
        if any([os.path.exists(os.path.join(visited_dir, "{}.json".format(idx))) for visited_dir in visited_dirs.split("||")]):
            continue

        completion = "Step 1: " + pred_dict['completion']
        instruction = prompt.format(problem=question, solution=pred_dict['gt_output'].replace("\n\n", "\n"), answer=completion.replace("\n\n", "\n"))

        # print("instruction: ", instruction)
        # import pdb; pdb.set_trace()

        while True:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": instruction,
                        }
                    ],
                    model="gpt-4o",
                )
            except (openai.APIConnectionError, openai.InternalServerError) as e:
                print(str(e))
                time.sleep(3)
                continue
            break

        item = pred_dict.copy()
        item['gpt4-output'] = chat_completion.choices[0].message.content
        item['gpt4-prompt'] = instruction
        save_path = os.path.join(save_dir, "{}.json".format(idx))
        with open(save_path, "w+") as f:
            json.dump(item, f, indent=4)
        cnt += 1
        print("cnt: ", cnt, "idx: ", idx)
        if cnt >= args.max_count_total:
            break

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default='qwen2-boxed-step')
    parser.add_argument("--visited_dirs", type=str, default='') # will skip the files in $visited_dirs
    parser.add_argument("--save_dir", type=str, default='./data_pipeline/generated')
    parser.add_argument("--remainder", type=int, default=0) # remainder
    parser.add_argument("--n_groups", type=int, default=1) # n_groups
    parser.add_argument("--json_files", type=str, default="./data_pipeline/predictions/qwen2-7b-instruct-temp0.8-top_p0.95_rep2_seed0-alpaca-group*.json")
    parser.add_argument("--max_count_per_question", type=int, default=1)
    parser.add_argument("--max_count_total", type=int, default=10000)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
