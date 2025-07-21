# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
import json
import torch
import pandas as pd
import lightning as L
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
  
    return dataset

def tokenizer_dataset_given_prompt(element, tokenizer, config, max_seq_length):
    encoded_prompt = tokenizer.encode(element['prompt'],
                                      max_length=max_seq_length,
                                      add_special_tokens=False,
                                      padding="max_length")

    encoded_prompt_and_response = tokenizer.encode(element['prompt_and_response'],
                                                   max_length=max_seq_length-1,
                                                   return_tensors="pt",
                                                   add_special_tokens=False,
                                                   padding="max_length")

    encoded_prompt_and_response = torch.cat((encoded_prompt_and_response.squeeze_(), torch.tensor([tokenizer.eos_token_id])))

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_prompt_and_response.clone()
    attention_mask = encoded_prompt_and_response.clone()

    if config["mask_prompt"]:
        labels[: len(encoded_prompt)] = config["ignore_index"]
    attention_mask[:(len(encoded_prompt_and_response))] = 1
    return {"input_ids": encoded_prompt_and_response.type(torch.int64), "labels": labels.type(torch.int64), "attention_mask": attention_mask.type(torch.int64)}
