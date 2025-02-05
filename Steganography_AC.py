import argparse
import torch.nn.functional as F
import os
import time
import torch
import random
import numpy as np
from sympy.logic.inference import valid
# from pandas.tests.arrays.boolean.test_comparison import dtype
# from torch.onnx.symbolic_opset9 import tensor
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import gc
import csv
from evaluate import load
import sys
import socket
from collections import Counter
import difflib
import sys
from math import log
from decimal import *
import objgraph
from collections import deque
from datasets import load_dataset
from itertools import islice


getcontext().prec = 150

class node(object):
    def __init__(self,index,word,weight,neighbour):
        self.index=index
        self.word=word
        self.weight=weight
        self.neighbour=neighbour

def find_connected_components(Nodes):
    visited = set()
    components = []
    for i in range(len(Nodes)):
        if i not in visited:
            component = []
            dfs(Nodes, i, visited, component)
            components.append(component)
    return components

def dfs(Nodes, i, visited, component):
    visited.add(i)
    component.append(i)
    for neighbor in Nodes[i].neighbour:
        if neighbor not in visited:
            dfs(Nodes, neighbor, visited, component)

def BFS_Forest(Forest_nodes,Nodes):
    start=Forest_nodes[0]

    queue = deque()
    visited = set()
    queue.append(start)
    queue.append(None)
    visited.add(start)
    result = []
    level = []
    while queue:
        node = queue.popleft()
        if node!=None:
            level.append(node)
            for neighbor in Nodes[node].neighbour:
                if neighbor not in Forest_nodes:
                    continue
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
            if  queue[0]==None:
                queue.popleft()
                result.append(level)
                level = []
                if queue:
                    queue.append(None)
        else:
            if queue:
                queue.append(None)
    return result,visited

def find_MWIS(Nodes):

    visited=[]
    Forest_nodes=[]
    for i in range(len(Nodes)):
        visited.append(i)
        index=0
        for nei in Nodes[i].neighbour:
            if nei in visited:
                index+=1
        if index<=1:
            Forest_nodes.append(i)

    # print(Forest_nodes)
    Layer=[]

    while Forest_nodes!=[]:

        layer_result,visited=BFS_Forest(Forest_nodes,Nodes)
        # print(visited)
        for node in visited:
            Forest_nodes.remove(node)
        # print(Forest_nodes)

        Layer.append(layer_result)
        # print(Layer)
    return Layer

def DP(subtree,Nodes):
    if len(subtree)==1:
        return subtree[0]

    sum_weight=[]
    sub_final=[]
    group_final=[]
    # print('Original')
    for item in subtree:
        temp=0
        for i in item:
            temp+=Nodes[i].weight
            # print(Nodes[i].word,Nodes[i].weight)
        sum_weight.append(temp)
    N=len(sum_weight)
    dp=[0]*(N+1)
    dp[0]=0
    dp[1]=sum_weight[0]
    # print(sum_weight)

    for k in range(2,N+1):
        # print(k)
        dp[k]=max(dp[k-1],sum_weight[k-1]+dp[k-2])

    k=N
    while k>=1:
        if dp[k]==dp[k-1]:
            k-=1
        else:
            group_final.append(k-1)
            k-=2
    # print('Final:')
    for i in group_final:
        for j in subtree[i]:
            sub_final.append(j)
            # print(Nodes[j].word, Nodes[j].weight)
    return sub_final

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
    if args.steganographic > 0:
        file_name = "./Address_glitch_steganography_data/{}/{}_steganography_{}_{}_top{}.tsv".format(
            args.disambiguation, model_name.split('/')[-1], args.encoding, args.disambiguation, args.top_k)
    else:
        file_name = './Address_glitch_steganography_data/{}_nonstega.tsv'.format(model_name.split('/')[-1])
    return model_name, file_name, prompts

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


def top_p_sampling(probabilities, p):
    """
    实现 top-p 采样 (nucleus sampling)

    :param probabilities: Tensor, 模型输出的概率分布
    :param p: float, 累积概率阈值 (0 < p <= 1)
    :return: Tensor, 被选中的索引
    """
    # Step 1: 降序排序
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

    # Step 2: 计算累积概率
    # cumulative_probs = torch.cumsum(sorted_probs, dtype=torch.float64, dim=-1)
    cumulative_probs = torch.tensor([torch.sum(sorted_probs[0][:i+1]) for i in range(sorted_probs.shape[1])], dtype= torch.float32).unsqueeze(0).to(model.device)
    # Step 3: 找到累积概率超过 p 的位置
    cutoff_index = torch.searchsorted(cumulative_probs.squeeze(0), p, right=True)

    # Step 4: 截断到满足 top-p 的 token
    selected_probs = sorted_probs[:,:cutoff_index + 1]
    selected_indices = sorted_indices[:,:cutoff_index + 1]

    return selected_probs, selected_indices

def get_valid_tokens_Consistency(probabilities):

    # for kind in ['top_p', 'top_k']:
    for k in [args.top_k, args.top_k * 2, args.top_k * 4, args.top_k * 8, args.top_k * 16]:
        topp_probabilities, top_indices = torch.topk(probabilities, k)

        # suffix = torch.arange(len(tokenizer.vocab), device = model.device).unsqueeze(1)
        repeated_tensor = inputs['input_ids'].repeat(top_indices.shape[1],1)
        candidate_tensor = torch.cat([repeated_tensor, top_indices.T], dim=1)
        del repeated_tensor
        torch.cuda.empty_cache()
        candidate_text = [tokenizer.decode(tensor, skip_special_tokens=True) for tensor in candidate_tensor]
    
        retokenized_tokenlists = [tokenizer(text, return_tensors="pt").to(model.device)['input_ids'] for text in candidate_text]
        # next_token_id = torch.multinomial(probabilities, num_samples=1).to(model.device)
        # current_length = candidate_tensor.shape[1]
        valid_tokenlist = []
        for i in range(top_indices.shape[1]):
            if torch.equal(retokenized_tokenlists[i].squeeze(0),candidate_tensor[i]):
                # if retokenized_tokenlists[i].squeeze(0)[-1] == candidate_tensor[i][-1]:
                valid_tokenlist.append(candidate_tensor[i][-1])
        del candidate_tensor
        torch.cuda.empty_cache()
        if valid_tokenlist != []:
            break
    

    return valid_tokenlist, top_indices.shape[1]

def get_valid_tokens_Basic(probabilities):
    topp_probabilities, top_indices = torch.topk(probabilities, args.top_k)
    keys_to_remove = set()
    id_word_dict = dict()
    for i in range(top_indices.shape[1]):
        id_word_dict[top_indices.squeeze(0)[i].item()] = tokenizer.decode(top_indices.squeeze(0)[i], skip_special_tokens=True)
    for id1, str1 in id_word_dict.items():
        for id2, str2 in id_word_dict.items():
            if id1 != id2 and str2.startswith(str1) and (id2 not in keys_to_remove): #str1是前缀，to be removed
                keys_to_remove.add(id1)
                # print(id1, str1, id2, str2)
    valid_tokenlist = list(set(id_word_dict.keys()) - keys_to_remove)
    # print(valid_tokenlist)
    return valid_tokenlist, top_indices.shape[1]

def get_valid_tokens_MWIS(probabilities):
    topp_probabilities, top_indices = torch.topk(probabilities, args.top_k)
    Nodes = []
    MWIS_Nodes = []
    id_word_dict = dict()
    for i in range(top_indices.shape[1]):
        id_word_dict[top_indices.squeeze(0)[i].item()] = tokenizer.decode(top_indices.squeeze(0)[i], skip_special_tokens=True)
    idlist = list(id_word_dict.keys())
    wordlist = list(id_word_dict.values())
    for i in range(len(wordlist)):
        neighbour = []
        for j in range(len(wordlist)):
            if i != j and (wordlist[i].startswith(wordlist[j]) or wordlist[j].startswith(wordlist[i])):
                neighbour.append(j)
        Nodes.append(node(i, wordlist[i], topp_probabilities.squeeze(0)[i].item(), neighbour))
    Layer = find_MWIS(Nodes)
    # print(Layer)
    final_index = []

    for subtree in Layer:
        sub_final = DP(subtree, Nodes)
        final_index.extend(sub_final)
        # print(final_index)
    valid_tokenlist = []

    for i in range(len(idlist)):
        if i in final_index:
            valid_tokenlist.append(idlist[i])
    # if len(final_index) != args.top_k:
    #     print('Original:')
    #     print(wordlist)
    #     print('Removing:')
    #     print([id_word_dict[i] for i in list(set(idlist) - set(valid_tokenlist))])

    return valid_tokenlist, top_indices.shape[1]

def get_valid_tokens_SyncPool(probabilities):
    topp_probabilities, top_indices = torch.topk(probabilities, args.top_k)
    Nodes = []

    id_word_dict = dict()
    for i in range(top_indices.shape[1]):
        id_word_dict[top_indices.squeeze(0)[i].item()] = tokenizer.decode(top_indices.squeeze(0)[i], skip_special_tokens=True)
    idlist = list(id_word_dict.keys())
    wordlist = list(id_word_dict.values())
    for i in range(len(wordlist)):
        neighbour = []
        for j in range(len(wordlist)):
            if i != j and (wordlist[i].startswith(wordlist[j]) or wordlist[j].startswith(wordlist[i])):
                neighbour.append(j)
        Nodes.append(node(i, wordlist[i], topp_probabilities.squeeze(0)[i].item(), neighbour))
    components = find_connected_components(Nodes)
    merged_tokenlist = []
    for item in components:
        merged_tokenlist.append([idlist[index] for index in item])

    return merged_tokenlist

def Arithmetic_encoding(valid_probabilities, current_min, current_max, secret_value):
    # print(torch.sum(valid_probabilities).item())
    value_interval = []
    for i in range(valid_probabilities.shape[0]):
        if i == 0:
            value_interval.append([torch.tensor(0.0, device=model.device), torch.sum(valid_probabilities[:i+1])])
        else:
            value_interval.append([value_interval[i-1][-1], torch.sum(valid_probabilities[:i+1])])
    # print(value_interval[-1][-1].item())
    Max_interval = current_max - current_min
    # print([current_min, current_max])
    # print(Max_interval)
    for i in range(len(value_interval)):
        temp = current_min + Decimal(value_interval[i][0].item()) * Max_interval
        value_interval[i][0] = temp
        temp = current_min + Decimal(value_interval[i][1].item()) * Max_interval
        value_interval[i][1] = temp
    # print(current_max - value_interval[-1][-1])
    selected_index = 'error'
    for i in range(len(value_interval)):
        if value_interval[i][-1] >= secret_value and value_interval[i][0] <= secret_value:
            selected_index = i
            current_min = value_interval[i][0]
            current_max = value_interval[i][-1]
            break

    # if selected_index == 'error':
    #     print(current_min, current_max, current_min)
    #     print(current_max - current_min)
    #     print(secret_value - current_min)
    #     print(secret_value - current_max)

    return selected_index, current_min, current_max

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--top_p', default = 0.01, type = float, required = False) #0.9, 0.99, 0.999, 0.9999
    parser.add_argument('--top_k', default = 128, type = int, required = False) #[16, 32, 64, ...]
    parser.add_argument('--bit_length', default = 128, type=int, required=False)
    parser.add_argument('--model_index', default = 0, type=int, required=False)  #0: EN, 1:JP, 2: CN
    parser.add_argument('--steganographic', default = 1, type = int, required=False)
    parser.add_argument('--encoding', default = 'AC', type=str, required=False)
    parser.add_argument('--disambiguation', default = 'None', type=str, required=False) #['None','Basic', 'MWIS', 'SyncPool', 'Consistency']
    parser.add_argument('--sample_num', default= 100, type=int, required=False)
    # parser.add_argument('--gpu_index', default = 0, type=int, required=False)

    # parser.add_argument('--top_k', default=64, type=int, required=False)
    # parser.add_argument('--seed', default=1999, type=int, required=False)
    args = parser.parse_args()
    print(args)

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name, file_name, prompts = switch_model(args.model_index)

    print(device)
    print(model_name, file_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, TOKENIZERS_PARALLELISM = True, padding_side='left')
    model.resize_token_embeddings(len(tokenizer.vocab))
    # top_k = round(len(tokenizer.vocab) * portion)
    MAX_NUM = Decimal(2 ** args.bit_length)

    for _ in range(args.sample_num):
        gc.collect()
        prompt = get_prompt(random.choice(prompts)['text'])
        # print(prompt)
        secret_bits = ''.join(str(random.randint(0, 1)) for _ in range(args.bit_length))

        secret_value = Decimal(int(secret_bits, 2))
        StartTime = time.time()
        # num += 1
        SUM_P = 0
        SUM_TOKEN = 0
        KLD_C = 0
        Entropy = 0
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        token_num = 0
        current_min = Decimal(0)
        current_max = Decimal(MAX_NUM)

        if args.steganographic <= 0:
            output_ids = model.generate(inputs['input_ids'], do_sample = True, max_length=random.randint( 10, 128), num_return_sequences=100)
            texts = []
            for i in range(output_ids.shape[0]):
                texts.append(tokenizer.decode(output_ids[i].squeeze(0), skip_special_tokens=True))
            perplexity = load("perplexity", module_type="metric")
            PPL = perplexity.compute(predictions=texts, model_id=model_name, device='cuda')['perplexities']
            file_exist = os.path.exists(file_name)

            with open(file_name, mode='a+', newline='') as file:
                writer = csv.writer(file, delimiter='\t')
                if not file_exist:
                    writer.writerow(
                        ['Non-stega text', 'PPL'])
                # next(writer)
                writer.writerows(list(zip(texts, PPL)))
            continue

        while True:
            if time.time() - StartTime > 120:
                break
            # if i%10 == 0:
            #     print(i)
            with torch.no_grad():
                outputs = model(inputs['input_ids'])
            temp_logits = outputs.logits[:,-1,:].to(torch.float64)
            probabilities = torch.nn.functional.softmax(temp_logits, dim=-1)

            # current_top_p = args.top_p
            Entropy += -torch.sum(probabilities * torch.log2(probabilities + 1e-10), dim=1).item()  # 加小值避免 log(0)

            if args.disambiguation == 'Consistency':
                valid_tokenlist, top_size = get_valid_tokens_Consistency(probabilities)
            elif args.disambiguation == 'Basic':
                valid_tokenlist, top_size = get_valid_tokens_Basic(probabilities)
            elif args.disambiguation == 'MWIS':
                valid_tokenlist, top_size = get_valid_tokens_MWIS(probabilities)
            elif args.disambiguation == 'None':
                topp_probabilities, top_indices = torch.topk(probabilities, args.top_k)
                valid_tokenlist = top_indices.squeeze(0).tolist()
            elif args.disambiguation == 'SyncPool':
                merged_tokenlist = get_valid_tokens_SyncPool(probabilities)

            if args.disambiguation != 'SyncPool':
                if len(valid_tokenlist) == 0:
                    print('No valid token!')
                    with open(file_name, mode='a+', newline='') as file:
                        writer = csv.writer(file, delimiter='\t')
                        # next(writer)
                        writer.writerow([False])
                    break
                valid_tokenlist = torch.tensor(valid_tokenlist, device=model.device)
                valid_probabilities_original = probabilities.squeeze(0)[valid_tokenlist]
                valid_probabilities = valid_probabilities_original / torch.sum(valid_probabilities_original)
                SUM_P += torch.sum(valid_probabilities_original).item()
                SUM_TOKEN += valid_tokenlist.shape[0]
                KLD_C += -log(torch.sum(valid_probabilities_original).item())
                if args.encoding == 'AC':
                    selected_index, current_min, current_max= Arithmetic_encoding(valid_probabilities, current_min, current_max, secret_value)
                if selected_index == 'error':
                    break
                token_num += 1
                # print(valid_tokenlist)
                next_token = valid_tokenlist[selected_index]

            elif args.disambiguation == 'SyncPool':
                merged_index = torch.tensor(list(range(len(merged_tokenlist))), device = model.device)
                merged_probabilities = []
                inner_probabilities = []
                P = 0
                for item in merged_tokenlist:
                    prob = 0
                    inner = []
                    for id in item:
                        prob += probabilities.squeeze(0)[id]
                        P += probabilities.squeeze(0)[id].item()
                        inner.append(probabilities.squeeze(0)[id])
                    merged_probabilities.append(prob)
                    inner_probabilities.append(torch.tensor(inner, device=model.device))

                # merged_tokenlist = torch.tensor(merged_tokenlist, device = model.device)
                # inner_probabilities = torch.tensor(inner_probabilities, device = model.device)
                merged_probabilities = torch.tensor(merged_probabilities, device = model.device)
                merged_probabilities = merged_probabilities / torch.sum(merged_probabilities)
                SUM_P += P
                SUM_TOKEN += args.top_k
                KLD_C += -log(P)
                token_num += 1
                if args.encoding == 'AC':
                    selected_index, current_min, current_max= Arithmetic_encoding(merged_probabilities, current_min, current_max, secret_value)
                if selected_index == 'error':
                    break
                # merged_tokenlist[selected_index]
                sampling_probabilities = inner_probabilities[selected_index] / torch.sum(inner_probabilities[selected_index])
                sampled_index = torch.multinomial(sampling_probabilities, num_samples=1).to(model.device)
                next_token = torch.tensor(merged_tokenlist[selected_index][sampled_index], device = model.device)

            inputs['input_ids'] = torch.cat((inputs['input_ids'], next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
            # print(secret_value - current_min)
            if secret_value - current_min < 1.0:
                break
            # print(inputs['input_ids'].squeeze(0).tolist())
            # text = tokenizer.decode(inputs['input_ids'].squeeze(0), skip_special_tokens=True)
            # print(text)
            # print(tokenizer([text], return_tensors="pt").to(model.device)['input_ids'].squeeze(0).tolist())
            # print(torch.equal(inputs['input_ids'].squeeze(0), tokenizer([text], return_tensors="pt").to(model.device)[
            #     'input_ids'].squeeze(0)))
            # 
            # print()
        if time.time() - StartTime > 120:
            continue

        if selected_index == 'error':
            print('float error, again!')
            continue

        EndTime = time.time()
        Total_time = EndTime - StartTime
        Avg_time = Total_time / token_num
        Bpw = args.bit_length / token_num
        SUM_P /= token_num
        SUM_TOKEN /= token_num
        KLD_C /= token_num
        Entropy /= token_num
        text = tokenizer.decode(inputs['input_ids'].squeeze(0), skip_special_tokens=True)
        if args.disambiguation == 'Consistency' or args.disambiguation == 'None':
            Disambiguation_result = torch.equal(inputs['input_ids'].squeeze(0), tokenizer([text], return_tensors="pt").to(model.device)['input_ids'].squeeze(0))
        else:
            Disambiguation_result = True

        # print(text)
        # perplexity = load("perplexity", module_type="metric")
        # PPL = perplexity.compute(predictions= [text], model_id = model_name, device = 'cuda')['mean_perplexity']
        # del perplexity
        torch.cuda.empty_cache()
        file_exist = os.path.exists(file_name)
        with open(file_name, mode='a+', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            if not file_exist:
                writer.writerow(['Disambiguation_result','Bpw','Utilization','KLD_C','Total_time','Avg_time','Sum_p','Sum_token','Prompt','Text'])
            # next(writer) 
            writer.writerow([str(Disambiguation_result), str(Bpw), str(Bpw/Entropy), str(KLD_C), str(Total_time), str(Avg_time), str(SUM_P), str(SUM_TOKEN), prompt, text])
        # 查看某个类型的对象数量
        # objgraph.show_most_common_types()
        gc.collect()


