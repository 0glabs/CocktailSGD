#!/usr/bin/env python

import torch
import torch.nn as nn
import argparse
from transformers import OPTForCausalLM, AutoConfig, AutoTokenizer
import os

def create_empty_opt(config):
    """Create an uninitialized OPT model to load checkpoint weights into."""
    _reset_parameters_linear = nn.Linear.reset_parameters
    def dummy(*args, **kwargs):
        pass
    nn.Linear.reset_parameters = dummy

    with torch.no_grad():
        model = OPTForCausalLM(config).eval()

    nn.Linear.reset_parameters = _reset_parameters_linear
    print("====== origin keys")
    for k in model.state_dict().keys():
        print(k)
    return model

def load_decentralized_checkpoint(model, checkpoint_path, n_stages=2, n_layer_per_stage=12):
    """Load the distributed checkpoint into the OPT model."""
    print(f"======= {len(model.model.decoder.layers)=}")
    print(f"======= {n_stages=} * {n_layer_per_stage}")
    assert n_stages * n_layer_per_stage >= len(model.model.decoder.layers)
    assert model.lm_head.weight.data is not model.model.decoder.embed_tokens.weight.data

    for i in range(n_stages):
        print(f'Loading stage {i}')
        checkpoint = torch.load(os.path.join(checkpoint_path, f'prank_{i}_checkpoint.pt'), map_location=torch.device("cpu"))
        print("====== checkpoint keys")
        for k in checkpoint.keys():
            print(k)

        if i == 0:
            _tmp = {k[len(f"0."):]: v for k, v in checkpoint.items() if k.startswith(f"0.")}
            model.model.decoder.embed_tokens.weight.data[:] = _tmp['embed_tokens.weight']

            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j+1}."):]: v for k, v in checkpoint.items() if k.startswith(f"{j+1}.")}
                if len(_tmp) == 0:
                    break
                model.model.decoder.layers[j].load_state_dict(_tmp)

        elif i == n_stages - 1:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]: v for k, v in checkpoint.items() if k.startswith(f"{j}.")}
                if 'lm_head.weight' in _tmp:
                    break
                model.model.decoder.layers[i * n_layer_per_stage + j].load_state_dict(_tmp)
            else:
                _tmp = {k[len(f"{n_layer_per_stage}."):]: v for k, v in checkpoint.items() if k.startswith(f"{n_layer_per_stage}.")}
            if len(_tmp) == 0:
                break
            model.model.decoder.final_layer_norm.weight.data[:] = _tmp['final_layer_norm.weight']
            model.model.decoder.final_layer_norm.bias.data[:] = _tmp['final_layer_norm.bias']
            model.lm_head.weight.data[:] = _tmp['lm_head.weight']

        else:
            for j in range(n_layer_per_stage):
                _tmp = {k[len(f"{j}."):]: v for k, v in checkpoint.items() if k.startswith(f"{j}.")}
                if len(_tmp) == 0:
                    break
                model.model.decoder.layers[i * n_layer_per_stage + j].load_state_dict(_tmp)

    return model



def convert(ckpt_path, save_path, n_stages = 1, n_layers = 24):
    config = AutoConfig.from_pretrained('facebook/opt-1.3b')
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
    model = create_empty_opt(config)
    load_decentralized_checkpoint(model, ckpt_path, n_stages=n_stages, n_layer_per_stage=n_layers)

    model.save_pretrained(save_path)
    config.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def main():
    parser = argparse.ArgumentParser(description='Convert OPT decentralized checkpoints to Hugging Face format')
    parser.add_argument('--ckpt-path', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--save-path', type=str, required=True, help='Path to save converted model')
    parser.add_argument('--n-stages', type=int, default=2, help='Pipeline group size')
    parser.add_argument('--n-layer-per-stage', type=int, default=12, help='Number of layers per pipeline stage')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    convert(args.ckpt_path, args.save_path, args.n_stages, args.n_layer_per_stage)
