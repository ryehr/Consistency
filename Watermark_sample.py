import argparse
import torch.nn.functional as F
import os
import time
import torch
import random
import numpy as np
# from pandas.tests.arrays.boolean.test_comparison import dtype
from torch.onnx.symbolic_opset9 import tensor
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import csv
from evaluate import load
from datasets import load_dataset
from itertools import islice
import math


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def get_prompt(initial_text):
    if args.model_index == 0:
        word_num = 0
        index = -1
        for i in range(len(initial_text)):
            if initial_text[i] in [' ', '\n', ',', '.', '?', '!', '？', '！', '。',';', '；', ':', '：']:
                word_num += 1
            if word_num >= 10:
                index = i
                break
        if index == -1:
            prompt = initial_text
        else:
            prompt = initial_text[:index]
    else:
        prompt = initial_text[:10]
    return prompt

def switch_model(model_index):
    if model_index == 0:
        model_name = "meta-llama/Llama-2-7b-hf"
        prompts = load_dataset("allenai/c4", "en", streaming=True)
        # portion = 0.2
    elif model_index == 1:
        model_name = "tokyotech-llm/Swallow-7b-hf"
        prompts = load_dataset("allenai/c4", "ja",  streaming=True)
        # portion = 0.2
    elif model_index == 2:
        model_name = "Qwen/Qwen2.5-7B"
        prompts = load_dataset("allenai/c4", "zh", streaming=True)
        # portion = 0.1
    prompts = list(islice(prompts['train'], 1000))

    if args.watermarking > 0:
        file_name = "./Address_glitch_watermark_data/{}_sample_watermark_period_{}_window_{}_length_{}.tsv".format(model_name.split('/')[-1], args.glitch_period, args.c_window, args.token_num)
    else:
        file_name = "./Address_glitch_watermark_data/{}_randomsample_watermark_period_{}_window_{}_length_{}.tsv".format(
            model_name.split('/')[-1], args.glitch_period, args.c_window, args.token_num)
    return model_name, file_name, prompts


def generate(inputs, glitch_removal):
    StartTime = time.time()
    new_token_num = 0
    glitch_after = None
    glitch_list = [[] for _ in range(args.token_num)]
    glitch_return = False

    # glitch_position = None
    while True:
        if time.time() - StartTime > 30:
            return None, None
        with torch.no_grad():
            outputs = model(inputs['input_ids'])
        temp_logits = outputs.logits[:, -1, :].to(torch.float64).squeeze()
        probabilities = torch.nn.functional.softmax(temp_logits, dim=-1)


        # print(temp_seed.item(), end=' ')
        if glitch_return == True:
            temp_logits[torch.tensor(glitch_list[new_token_num], dtype = torch.long)] = -float('inf')
            glitch_return = False
            glitch_after = None
            probabilities = torch.nn.functional.softmax(temp_logits, dim=-1)

        # max_values, indices = torch.topk(probabilities, 12800)

        temp_seed = inputs['input_ids'].squeeze(0)[-args.c_window:].min()
        generator.manual_seed(temp_seed.item())
        R = torch.rand(vocab_size, generator = generator, device = model.device, dtype=torch.float64)
        R = torch.log(R)/probabilities
        if args.watermarking > 0:
            next_token = torch.argmax(R).to(model.device)
        else:
            next_token = torch.multinomial(probabilities, num_samples=1).to(model.device).squeeze(0)

        inputs['input_ids'] = torch.cat((inputs['input_ids'], next_token.unsqueeze(0).unsqueeze(0)), dim=-1)

        if glitch_removal == True:
            temp_text = tokenizer.decode(inputs['input_ids'].squeeze(0), skip_special_tokens=True)
            retokenlist = tokenizer(temp_text, return_tensors="pt").to(model.device)['input_ids']

            if torch.equal(inputs['input_ids'], retokenlist) == True:
                # if glitch_after != None:
                #     print('Auto removal')
                glitch_after = None

            else: #torch.equal(inputs['input_ids'], retokenlist) == False:
                # print(new_token_num)
                if glitch_after == None:
                    glitch_after = 0
                    glitch_list[new_token_num].append(next_token.squeeze(0))

                if glitch_after != None:
                    if glitch_after < args.glitch_period and new_token_num + 1 < args.token_num:
                        glitch_after += 1
                    else:
                        inputs['input_ids'] = inputs['input_ids'][:, :-(glitch_after + 1)]

                        new_token_num = new_token_num - 1 - glitch_after
                        glitch_return = True
                        # print('Return!!!!')
        new_token_num += 1
        # print(new_token_num)
        if new_token_num >= args.token_num:
            break
    text = tokenizer.decode(inputs['input_ids'].squeeze(0), skip_special_tokens=True)

    # print(inputs['input_ids'].squeeze(0))
    return text, inputs['input_ids'].squeeze(0)


def detect(key,text,real_tokenlist):
    if key == 'text':
        tokenlist = tokenizer(text, return_tensors="pt").to(model.device)['input_ids'].squeeze(0)
    elif key == 'tokens':
        tokenlist = real_tokenlist
    # print(tokenlist)

    R_temp = 0
    # print(tokenlist.size(0))
    for i in range(1, tokenlist.size(0)):
        temp_seed = tokenlist[0:i][-args.c_window:].min()
        generator.manual_seed(temp_seed.item())
        R = torch.rand(vocab_size, generator=generator, device=model.device, dtype=torch.float64)
        # print(tokenlist[0:i][-args.c_window + 1:].sum().item() + tokenlist[i].item())
        add = -math.log(1 - R[tokenlist[i]])
        # print(add)
        R_temp += add
    R_final = 1/math.sqrt(tokenlist.size(0)) * R_temp - math.sqrt(tokenlist.size(0))

    return R_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--temperature', default = 1.0, type = float, required = False)
    # parser.add_argument('--top_k', default=64, type=int, required=False)
    # parser.add_argument('--seed', default=1999, type=int, required=False)
    parser.add_argument('--model_index', default = 0, type=int, required=False)  #0: EN, 1:JP, 2: CN
    parser.add_argument('--token_num', default = 200, type = int, required = False)
    parser.add_argument('--c_window',default = 5, type = int, required = False)
    parser.add_argument('--sample_num',default = 100, type = int, required = False)
    parser.add_argument('--glitch_period', default = 2, type=int, required=False) # 最小为0，表示立即执行
    parser.add_argument('--watermarking', default=1, type=int, required=False)  # 0表示无水印
    # parser.add_argument('--glitch_removal', default = 1, type = int, required = False) # 0表示不消除glitch
    args = parser.parse_args()
    print(args)
    print(os.getcwd())
    print('GPU状态为：',torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name, file_name, prompts = switch_model(args.model_index)
    print(model_name, file_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, TOKENIZERS_PARALLELISM = True, padding_side='left')
    model.resize_token_embeddings(len(tokenizer.vocab))
    generator = torch.Generator(device=model.device)
    vocab_size = len(tokenizer.get_vocab())

    for _ in range(args.sample_num):
        prompt = get_prompt(random.choice(prompts)['text'])
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        temp_text = tokenizer.decode(inputs['input_ids'].squeeze(0), skip_special_tokens=True)
        retokenlist = tokenizer(temp_text, return_tensors="pt").to(model.device)['input_ids']
        if torch.equal(inputs['input_ids'], retokenlist) == False:
            continue
        constant_seed = torch.randint(1, 99999999, (1,)).to(model.device)
        generator.manual_seed(constant_seed.item())
        random_sequence = torch.randint(1, 99999999, (1000,), generator=generator, device=model.device)

        glitch_text, real_tokenlist = generate(inputs, glitch_removal=False)
        if glitch_text == None:
            continue
        # print(glitch_text)
        R_list_glitch_text = detect(key='text', text=glitch_text, real_tokenlist=real_tokenlist)
        R_list_real_token = detect(key='tokens', text=glitch_text, real_tokenlist=real_tokenlist)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        glitch_removal_text, glitch_removal_tokenlist = generate(inputs, glitch_removal=True)
        if glitch_removal_text == None:
            continue
        # print(glitch_removal_text)
        R_removal_text = detect(key='text',text = glitch_removal_text, real_tokenlist=glitch_removal_tokenlist)
        R_removal_token = detect(key='tokens',text = glitch_removal_text, real_tokenlist=glitch_removal_tokenlist)
        # print(R_list_glitch_text, R_list_real_token, R_removal_text, R_removal_token)
        file_exist = os.path.exists(file_name)
        with open(file_name, mode='a+', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            if not file_exist:
                writer.writerow(['Constant_seed','Z_glitch_final', 'Z_removal_final', 'Glitch_text', 'Glitch_removal_text'])
            # next(writer)
            writer.writerow([str(constant_seed.item()), str(R_list_glitch_text), str(R_removal_text), glitch_text, glitch_removal_text ])

        pass
