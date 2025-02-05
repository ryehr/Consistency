import argparse
import torch.nn.functional as F
import os
import time
import torch
import random
import pandas as pd
import numpy as np
# from pandas.tests.arrays.boolean.test_comparison import dtype
from torch.onnx.symbolic_opset9 import tensor
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import csv
from evaluate import load
from datasets import load_dataset
from itertools import islice
import math




def Attack(text):
    tokens = tokenizer([text], return_tensors="pt").to(model.device)['input_ids']
    # print(tokens.size(1))
    for i in range(tokens.size(1)-1):
        if random.random() > args.attack_probability:
            continue
        with torch.no_grad():
            outputs = model(tokens[:i+1])
        temp_logits = outputs.logits[:, -1, :].to(torch.float64).squeeze()
        probabilities = torch.nn.functional.softmax(temp_logits, dim=-1)
        Token_replace = torch.multinomial(probabilities, num_samples=1).to(model.device)
        tokens[0][i+1] = Token_replace
    # print(tokens.size(1))
    return tokenizer.decode(tokens.squeeze(0), skip_special_tokens=True)

def detect_logits(text):

    tokenlist = tokenizer(text, return_tensors="pt").to(model.device)['input_ids'].squeeze(0)
    if 'window_0' in args.input_file:
        hash_window = 0
    elif 'window_1' in args.input_file:
        hash_window = 1
    elif 'window_4' in args.input_file:
        hash_window = 4

    green_num = 0
    Z_value_list = []
    for i in range(1, tokenlist.size(0)):
        if hash_window == 0:
            temp_green_list = constant_green_list
        elif hash_window > 0:
            temp_seed = tokenlist[0:i][-hash_window:].min()
            generator.manual_seed(temp_seed.item())
            temp_green_list = torch.randperm(vocab_size, generator=generator, device=model.device)[:int(vocab_size * 0.5)]
        # print(hash_window)
        # print(temp_green_list)
        if tokenlist[i] in temp_green_list:
            green_num += 1
        Z_value = (green_num - (i+1) * 0.5) / ((i+1) * 0.5 * (1 - 0.5)) ** 0.5
        Z_value_list.append(Z_value)

    return Z_value_list[-1]


def detect_sample(text):

    tokenlist = tokenizer(text, return_tensors="pt").to(model.device)['input_ids'].squeeze(0)
    R_temp = 0
    c_window = 5
    # print(tokenlist.size(0))
    for i in range(1, tokenlist.size(0)):
        temp_seed = tokenlist[0:i][-c_window:].min()
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


    parser.add_argument('--input_file', default = 'Llama-2-7b-hf_logits_watermark_priority_2.0_ratio_0.5_period_2_window_0_length_200.tsv', type = str, required=False)
    parser.add_argument('--attack_probability', default = 0.0, type = float, required=False)

    args = parser.parse_args()
    print(args)

    print('GPU状态为：',torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if 'Llama' in args.input_file:
        model_name = "meta-llama/Llama-2-7b-hf"

    elif 'Swallow' in args.input_file:
        model_name = "tokyotech-llm/Swallow-7b-hf"

    elif 'Qwen' in args.input_file:
        model_name = "Qwen/Qwen2.5-7B"


    print(model_name, args.input_file)
    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, TOKENIZERS_PARALLELISM = True, padding_side='left')
    model.resize_token_embeddings(len(tokenizer.vocab))
    generator = torch.Generator(device=model.device)
    vocab_size = len(tokenizer.get_vocab())
    Texts_glitch = pd.read_csv('./Address_glitch_watermark_data/' + args.input_file, sep='\t', keep_default_na=False)['Glitch_text'].tolist()
    Texts_removal = pd.read_csv('./Address_glitch_watermark_data/' + args.input_file, sep='\t', keep_default_na=False)['Glitch_removal_text'].tolist()
    output_file = './Address_glitch_watermark_data/Attack/{}_attack_{}.tsv'.format(args.input_file, args.attack_probability)
    for i in range(len(Texts_glitch)):
        attacked_glitch_text = Attack(Texts_glitch[i])
        attacted_removal_text = Attack(Texts_removal[i])
        constant_seed = int(pd.read_csv('./Address_glitch_watermark_data/' + args.input_file, sep='\t', keep_default_na=False)['Constant_seed'][i])
        # print(constant_seed)
        generator.manual_seed(constant_seed)
        random_sequence = torch.randint(1, 99999999, (1000,), generator=generator, device=model.device)
        constant_green_list = torch.randperm(vocab_size, generator=generator, device=model.device)[
                              :int(vocab_size * 0.5)]

        if '_logits_' in args.input_file:
            Z_glitch_final = detect_logits(attacked_glitch_text)
            Z_removal_final = detect_logits(attacted_removal_text)
        elif '_sample_' in args.input_file:
            Z_glitch_final = detect_sample(attacked_glitch_text)
            Z_removal_final = detect_sample(attacted_removal_text)
        # print(Z_glitch_final, Z_removal_final)

        file_exist = os.path.exists(output_file)
        with open(output_file, mode='a+', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            if not file_exist:
                writer.writerow(
                    ['Constant_seed', 'Z_glitch_final', 'Z_removal_final', 'Glitch_text', 'Glitch_removal_text'])
            # next(writer)
            writer.writerow(
                [str(constant_seed), str(Z_glitch_final), str(Z_removal_final), attacked_glitch_text, attacted_removal_text])

    pass