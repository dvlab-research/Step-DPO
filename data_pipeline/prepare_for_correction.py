import argparse
import json
import glob
import jsonlines
import re

def main(args):
    save_file = args.save_file
    generated_files = sorted(glob.glob(args.generated_files))

    invalid_cnt0 = 0
    invalid_cnt1 = 0
    invalid_cnt2 = 0
    with jsonlines.open(save_file, "w") as f:
        for json_file in generated_files:

            with open(json_file) as ff:
                item = json.load(ff)

            correctness = item['gpt4-output'].lower()
            correctness = correctness.split("final decision")[-1].split("summary decision:")[-1].strip()
            if not any([x in correctness for x in ['neutral', 'incorrect']]):
                invalid_cnt0 += 1
                continue
            
            step_num = correctness.split("neutral")[0].split("incorrect")[0]
            step_num = step_num.split("\n")[-1].split(";")[-1]
            if step_num.count("step") > 1:
                invalid_cnt1 += 1
                continue
            step_num = step_num.split("step")[-1].split(":")[0]
            try:
                step_num = int(step_num.strip())
            except:
                # import pdb; pdb.set_trace()
                invalid_cnt2 += 1
                continue

            if 'alpaca' in args.prompt:
                prompt = item['prompt'].split("### Instruction:")[1].split("### Response:")[0].strip()
                prefix = item['prompt'].split("### Response:")[-1].lstrip()
            elif 'qwen2-boxed' in args.prompt:
                prompt = item['prompt'].split("<|im_start|>user\n")[1].split("\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>")[0].strip()
                prefix = item['prompt'].split("<|im_start|>assistant\n")[-1].lstrip()
            else:
                raise NotImplementedError("Prompt {} is not supported currently".format(args.prompt))

            completion = prefix + item['completion']
            # pred_answer = completion.split("The answer is:")[-1].strip()
            type = item['type']
            
            if completion.count("Step {}:".format(step_num)) == 0:
                continue

            prefix = completion.split("Step {}:".format(step_num))[0] + "Step {}:".format(step_num)
            
            new_item = {
                'idx': "n/a",
                'instruction': prompt,
                'prefix': prefix.replace("Let's think step by step.\n", ""),
                'output': completion.replace(prefix, ""),
                'gt_output': item['gt_output'],
                'answer': item['prompt_answer'],
                'step_num': step_num,
                'input': "",
                'type': type,
                'ori_filepath': item['path'] if 'path' in item else 'n/a',
            }
            f.write(new_item)

    print("invalid_cnt0: ", invalid_cnt0)
    print("invalid_cnt1: ", invalid_cnt1)
    print("invalid_cnt2: ", invalid_cnt2)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default='qwen2-boxed-step')
    parser.add_argument("--save_file", type=str, default='./data_pipeline/continue_from_incorrect.jsonl')
    parser.add_argument("--generated_files", type=str, default="./data_pipeline/generated/*.json")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
