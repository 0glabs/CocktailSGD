import os
import re
import torch
import json
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import random
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from comm.comm_utils import *


from itertools import islice
from random import randint

SHOW_DATA = int(os.environ.get('SHOW_DATA', 0))


class StreamDatasetList(IterableDataset):
    def __init__(self, task_names, datasets, sample_probs, tokenizer, seq_length=1024, print_sample_every_n=64,):
        
        self.task_names = task_names
        self.datasets = datasets
        self.sample_probs = sample_probs
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.print_sample_every_n = print_sample_every_n
        
        self.it = None
        
    def state_dict(self):
        return {}
    
    def load_state_dict(self, state_dict):
        pass
        
    def get_sequence(self):
        
        iterators = [cycle(d.get_sequence()) for d in self.datasets]
        prob_ths = np.cumsum([p / sum(self.sample_probs) for p in self.sample_probs])
        
        global_i = 0
        
        while True:
            
            p = random.random()
            
            for task_name, it, th in zip(self.task_names, iterators, prob_ths):
                if p < th:
                    
                    inputs = next(it)
                    
                    if inputs['input_ids'].size(0) != self.seq_length:
                        print('!!', inputs['input_ids'].shape)
                        continue
                    
                    if SHOW_DATA:
                        if global_i % self.print_sample_every_n == 0:
                            print(p, th)
                            print(f"**{task_name}**:", self.tokenizer.decode(inputs['input_ids']))
                        
                    yield inputs
                    global_i += 1
                    break
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it
    
    
def name_to_dataset(task, tokenizer, args, data_path):

    if task != '':
        if task == 'natural_instructions' or task == 'ni':
            from .natural_instructions import StreamDataset
            dataset = StreamDataset(data_path, tokenizer, args.seq_length)
        elif task == 'p3':
            from .p3 import StreamDataset
            data = load_dataset(data_path, split="train").shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'natural_instructions_dehelm' or task == 'ni_dehelm':
            from .natural_instructions_dehelm import StreamDataset
            dataset = StreamDataset(data_path, tokenizer, args.seq_length)
        elif task == 'p3_dehelm':
            from .p3 import StreamDataset
            data = load_dataset(data_path, split='train').shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'pile':
            from .pile import StreamDataset
            data = load_dataset(data_path, split="train").shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'rp_sample':
            from .pile import StreamDataset
            data = load_dataset(data_path, split="train").shuffle(seed=args.seed).with_format("torch")
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'cot':
            from .cot import StreamDataset
            dataset = StreamDataset(data_path, tokenizer, args.seq_length)
        else:
            raise NotImplementedError(f"Task {task} is not supported.")

    return dataset

def name_to_eval(data_path, tokenizer, args):
    task = args.task_name

    if task != '':
        if task == 'natural_instructions' or task == 'ni':
            from .natural_instructions import StreamDataset
            dataset = StreamDataset(data_path, tokenizer, args.seq_length)
        elif task == 'p3':
            from .p3 import StreamDataset
            data = load_dataset(data_path, split="validation").shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'natural_instructions_dehelm' or task == 'ni_dehelm':
            from .natural_instructions_dehelm import StreamDataset
            dataset = StreamDataset(data_path, tokenizer, args.seq_length)
        elif task == 'p3_dehelm':
            from .p3 import StreamDataset
            data = load_dataset(data_path, split='validation').shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'pile':
            from .pile import StreamDataset
            data = load_dataset(data_path, split="validation").shuffle(seed=args.seed)
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'rp_sample':
            from .pile import StreamDataset
            data = load_dataset(data_path, split="validation").shuffle(seed=args.seed).with_format("torch")
            dataset = StreamDataset(data, tokenizer, args.seq_length)
        elif task == 'cot':
            from .cot import StreamDataset
            dataset = StreamDataset(data_path, tokenizer, args.seq_length)
        else:
            raise NotImplementedError(f"Task {task} is not supported.")

    return dataset

def name_to_dataset_eval(task, tokenizer, args):
    evaluation_data = args.evaluation_data
    task = args.task_name

    if task != '':
        if task == 'pile':
            from .pile import StreamDataset
            data = load_dataset(data_path, split="validation", streaming=True)
            dataset = StreamDataset(data, tokenizer, args.seq_length, cycling=False)
        elif task in [""]:
            from .pile import StreamDataset
            data = load_dataset("json", data_files=task, split="train", streaming=True) # jsonl file default is "train"
            dataset = StreamDataset(data, tokenizer, args.seq_length, cycling=False)
        else:
            from .pile import StreamDataset
            data = load_dataset(task, split="validation", streaming=True)
            dataset = StreamDataset(data, tokenizer, args.seq_length, cycling=False)
        
    return dataset

    
def get_train_data_loader(args, tokenizer, num_workers=1, state_dict=None):
    
    task_list = args.task_name.split(',')
    data_path = args.data_path.split(",")
    if len(task_list) != len(data_path):
        raise ValueError("task_name and data_path should have the same length")

    task_names = []
    datasets = []
    probs = []
    
    print('data_utils: parse task_list')
    
    for i, task in enumerate(task_list):
        if ':' in task:
            task, prob = task.strip().split(':')
            prob = float(prob)
        else:
            task = task.strip()
            prob = 1.0
            
        dataset = name_to_dataset(task, tokenizer, args, data_path[i])
            
        print('data_utils:', task, prob)
    
        task_names.append(task)
        datasets.append(dataset)
        probs.append(prob)
    
    stream_dataset = StreamDatasetList(
        task_names, datasets, probs,
        tokenizer=tokenizer, seq_length=args.seq_length)
    
    if state_dict is not None:
        stream_dataset.load_state_dict(state_dict)
    
    train_data_loader = torch.utils.data.DataLoader(stream_dataset,
                                                    batch_size=args.batch_size * args.data_group_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    prefetch_factor=16,
                                                    collate_fn=None)
    
    print('data_utils: get train_data_loader')
    
    return train_data_loader


def get_eval_data_loader(args, tokenizer, num_workers=1, state_dict=None):
    
    task_list = args.task_name.split(',')
    task_names = []
    datasets = []
    probs = []
    
    evaluation_data = args.evaluation_data

    if evaluation_data is None:
        return None

    if not os.path.isdir(evaluation_data):
        raise RuntimeError("Evaluation data path should be a folder.")
    
    dataset = name_to_eval(evaluation_data, tokenizer, args)
    
    train_data_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    drop_last=True,
                                                    num_workers=num_workers,
                                                    pin_memory=True,
                                                    collate_fn=None)
    
    return train_data_loader

