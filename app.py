import numpy as np

import gradio as gr
import re
import bleach
import sys
import os
import argparse
import torch
import transformers
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPImageProcessor

def parse_args(args):
  parser = argparse.ArgumentParser(description='LISA chat')
  parser.add_argument('--model_path_or_name', default='')
  parser.add_argument('--save_path', default='/data/step_dpo_history')
  parser.add_argument('--load_in_8bit', action='store_true', default=False)
  parser.add_argument('--load_in_4bit', action='store_true', default=False)
  return parser.parse_args(args)

args = parse_args(sys.argv[1:])
os.makedirs(args.save_path, exist_ok=True)

# Create model
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path_or_name)
model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path_or_name, torch_dtype=torch.bfloat16, device_map="auto")

# Gradio
examples = [
    ['Suppose that $h(x)=f^{-1}(x)$. If $h(2)=10$, $h(10)=1$ and $h(1)=2$, what is $f(f(10))$?'],
]
output_labels = ['Output']

title = 'Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs'

description = """
<font size=3>

This is the online demo of **Qwen2-7B-Instruct-Step-DPO**. \n

It is obtained by performing **Step-DPO** on **Qwen2-7B-Instruct**, with as few as **10K data and hundreds of training steps**. \n

**Step-DPO** improves the mathematical reasoning of **Qwen2-7B-Instruct** significantly, from **53.0\%** to **58.6\%** on MATH, and **85.5\%** to **87.9\%** on GSM8K. \n
Besides, **Qwen2-72B-Instruct-Step-DPO** achieves **70.8\%** on MATH and **94.0\%** on GSM8K, **outperforming GPT-4-1106, Gemini-1.5-Pro, and Claude-3-Opus**.

Code, models, data are available at [GitHub](https://github.com/dvlab-research/Step-DPO).

Hope you can enjoy our work!
</font>
"""

article = """
<p style='text-align: center'>
<a href='https://arxiv.org/pdf/2406.18629' target='_blank'>
Preprint Paper
</a>
\n
<p style='text-align: center'>
<a href='https://github.com/dvlab-research/Step-DPO' target='_blank'>   Github Repo </a></p>
"""


## to be implemented
def inference(input_str):

    ## filter out special chars
    input_str = bleach.clean(input_str)

    # print("input_str: ", input_str, "input_image: ", input_image)
    print("input_str: ", input_str)

    # ## input valid check
    # if not re.match(r'^[A-Za-z ,.!?\'\"]+$', input_str) or len(input_str) < 1:
    #     output_str = '[Error] Invalid input: ', input_str
    #     # output_image = np.zeros((128, 128, 3))
    #     ## error happened
    #     output_image = cv2.imread('./resources/error_happened.png')[:,:,::-1]
    #     return output_image, output_str

    # Model Inference
    # conv = get_default_conv_template("vicuna").copy()
    # conv.messages = []

    prompt = input_str + "\nPlease reason step by step, and put your final answer within \\boxed{{}}." #input("Please input your prompt: ")
    # prompt = DEFAULT_IMAGE_TOKEN + " " + prompt
    # replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
    # replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
    # prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    text_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return text_output


demo = gr.Interface(
    inference,
    inputs=[
        gr.Textbox(
            lines=1, placeholder=None, label='Math Problem'),
    ],
    outputs=[
        gr.Textbox(
            lines=1, placeholder=None, label='Text Output'),        
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
    allow_flagging='auto',
    flagging_dir=args.save_path)

demo.queue()

demo.launch(server_name='0.0.0.0', show_error=True)
