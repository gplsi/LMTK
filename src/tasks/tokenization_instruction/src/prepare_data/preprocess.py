# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
from abc import abstractmethod
from functools import partial
from typing import Dict, List
import os
import pandas as pd
import datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from torch import Tensor
from prepare_data.prompts import PromptStyle
import lightning as L
import json
from datasets import Dataset
from datetime import datetime

def createConversation(row):
    conversation = [{'role': 'system', 'content': row['system']}]

    for i in range(len(row['input'])):
        conversation.append({'role': 'user', 'content': row['input'][i]})

        if i < len(row['input']) - 1:
            conversation.append({'role': 'assistant', 'content': row['assistance'][i]})

    return conversation

def createPrompt(conversation, tokenizer):
    date_string = datetime.today().strftime('%Y-%m-%d')
    prompt = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True,
    date_string=date_string
    )

    return prompt

def tokenizer_dataset_multiTurn(dir, tokenizer, config, max_seq_length=2048):
    dir = os.path.join(os.getcwd(), dir)

    print("Loading dataset from folder: ", dir)
    
    dataset = pd.DataFrame(columns=['prompt', 'prompt_and_response'])

    for file in os.listdir(dir):
        print("Loading file: ", file)
        with open(os.path.join(dir, file), encoding="UTF8") as f:
            data = json.load(f)

        for element in data:
            conversation = createConversation(element)
            prompt = createPrompt(conversation, tokenizer)
            prompt_and_response = prompt + element['target'] + '<|im_end|>'
            dataset = pd.concat([dataset, pd.DataFrame({'prompt': [prompt], 'prompt_and_response': [prompt_and_response]})], axis=0)

    print('Tokenizing dataset...')

    dataset = Dataset.from_pandas(dataset)

    dataset = dataset.map(
        tokenizer_dataset_given_prompt,
        remove_columns=['prompt', 'prompt_and_response'],
        fn_kwargs={'tokenizer': tokenizer, 'config': config, 'max_seq_length': max_seq_length}
    )
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset

def tokenizer_dataset(dir, tokenizer, config, max_seq_length=2048):
    dir = os.path.join(os.getcwd(), dir)
    
    dataset = pd.DataFrame(columns=['system', 'instruction', 'output'])

    for file in os.listdir(dir):
        print(file)
        with open(os.path.join(dir, file)) as f:
            data = json.load(f)

        #systemPrompt = data[0]['system']
        #auxData = pd.DataFrame(data['prompts'])
        #auxData['system'] = systemPrompt
        auxData = pd.DataFrame(data)
        auxData.rename(columns={'input': 'instruction', 'target': 'output'}, inplace=True)
        dataset = pd.concat([dataset, auxData], axis=0)

    dataset = datasets.Dataset.from_pandas(dataset)

    dataset = dataset.map(
        tokenizer_dataset_with_prompt,
        remove_columns=['instruction', 'output', 'system'],
        fn_kwargs={'tokenizer': tokenizer, 'config': config, 'max_seq_length': max_seq_length}
    )
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset

def get_dataloader(dir, tokenizer, config, max_seq_length=2048, batch_size=4, shuffle=True, seed=1234, rank=1):
    dir = os.path.join(os.getcwd(), dir)
    list_files = os.listdir(dir)
    df_all = pd.DataFrame()
    for file in list_files[0:1]:
        df_temp = pd.read_excel(os.path.join(dir, file))
        df_temp.rename(columns={'inputs': 'instruction', 'targets': 'output'}, inplace=True)
        df_all = pd.concat([df_all, df_temp], axis=0)
    dataset = datasets.Dataset.from_pandas(df_temp)
    dataset = dataset.map(tokenizer_dataset_with_prompt, remove_columns=['instruction', 'output'],
                          fn_kwargs={'tokenizer': tokenizer, 'config': config, 'max_seq_length': max_seq_length})
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    print("rank_dataloader:", rank)
    sampler = DistributedSampler(dataset, rank=rank, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        #shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=4,
        sampler=sampler,
        collate_fn=get_sft_collate_fn(max_seq_length=max_seq_length, pad_id=0, ignore_index=-100),
    )

def tokenizer_dataset_given_prompt(element, tokenizer, config, max_seq_length):
    encoded_prompt = tokenizer.encode(element['prompt'], max_length=max_seq_length, add_special_tokens=False)
    encoded_prompt_and_response = tokenizer.encode(element['prompt_and_response'], max_length=max_seq_length-1, return_tensors="pt", add_special_tokens=False)
    encoded_prompt_and_response = torch.cat((encoded_prompt_and_response.squeeze_(), torch.tensor([tokenizer.eos_token_id])))

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_prompt_and_response.clone()
    attention_mask = encoded_prompt_and_response.clone()

    if config["mask_prompt"]:
        labels[: len(encoded_prompt)] = config["ignore_index"]
    attention_mask[:(len(encoded_prompt_and_response))] = 1
    return {"input_ids": encoded_prompt_and_response.type(torch.int64), "labels": labels.type(torch.int64), "attention_mask": attention_mask.type(torch.int64)}

def tokenizer_dataset_with_prompt(sample, tokenizer, config, max_seq_length):
    prompt_style = PromptStyle.from_name(config["prompt_style"])
    prompt = prompt_style.apply(prompt=sample, **sample)
    prompt_and_response = prompt + sample["output"]
    encoded_prompt = tokenizer.encode(prompt, max_length=max_seq_length)
    encoded_prompt_and_response = tokenizer.encode(prompt_and_response, max_length=max_seq_length-1, return_tensors="pt")
    encoded_prompt_and_response = torch.cat((encoded_prompt_and_response.squeeze_(), torch.tensor([tokenizer.eos_token_id])))

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_prompt_and_response.clone()
    attention_mask = encoded_prompt_and_response.clone()

    if config["mask_prompt"]:
        labels[: len(encoded_prompt)] = config["ignore_index"]
    attention_mask[:(len(encoded_prompt_and_response))] = 1
    return {"input_ids": encoded_prompt_and_response.type(torch.int64), "labels": labels.type(torch.int64), "attention_mask": attention_mask.type(torch.int64)}

def get_sft_collate_fn(max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100):
    """Returns the collate function for supervised finetuning (needed in the DataLoader).

    The collate function gets a list of dicts with keys `input_ids` and `labels`.
    It returns a dict with batched `input_ids` and `labels`. Also pads short sequences to the longest element in
    the batch. Optionally truncates all sequences to the specified maximum length.
    """
    return partial(_sft_collate_fn, max_seq_length=max_seq_length, pad_id=pad_id, ignore_index=ignore_index)

def _sft_collate_fn(
    samples: List[Dict[str, Tensor]], max_seq_length: int = -1, pad_id: int = 0, ignore_index: int = -100
) -> Dict[str, Tensor]:

    batched = {}
    for key in ("input_ids", "labels", "attention_mask"):
        pad_value = pad_id if key == "input_ids" else ignore_index

        # Pad right based on the longest sequence
        batched[key] = torch.nn.utils.rnn.pad_sequence(
            [sample[key] for sample in samples], batch_first=True, padding_value=pad_value
        )

        # Truncate if needed
        if max_seq_length > 0:
            batched[key] = batched[key][:, :max_seq_length]

    return batched
