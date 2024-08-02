import json
import os
import sys
from pprint import pprint

import torch
from torch import nn

from modules.kdd_models import KDDTransformerEncoderClassification, KDDTransformerEncoderImputation
from modules.kdd_original_models import KDDOriginalTransformerEncoderImputation, \
    KDDOriginalTransformerEncoderClassification


def load_create_classification_model(config: dict, num_classes: int) -> (nn.Module, dict):
    if config['pretrain_model_path']:
        sys.exit("Pretrained model loading is not implemented yet.")
        with open(os.path.join(config['pretrain_model_path'], 'kdd_model_config.json'), 'r') as f:
            model_config = json.load(f)

        model = KDDTransformerEncoderClassification(**model_config)

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)

        # Load the pre-trained state dict
        pretrained_state_dict = torch.load(os.path.join(config['pretrain_model_path'], 'kdd_model.pth'), map_location=device)

        # Remove the 'output' layer from the pre-trained state dict
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('output')}

        # Load the modified state dict
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict, strict=False)

        print("Pretrained model loaded successfully (without output layer).")

    else:
        if config['kdd_model']['conv_config']:
            conv_config = config[config['kdd_model']['conv_config']]
        else:
            conv_config = None

        model_config = {
            'feat_dim': config['kdd_model']['feat_dim'],
            'max_len': config['kdd_model']['max_seq_len'],
            'd_model': config['kdd_model']['d_model'],
            'n_heads': config['kdd_model']['n_heads'],
            'n_layers': config['kdd_model']['n_layers'],
            'dim_feedforward': config['kdd_model']['dim_feedforward'],
            'emb_dropout': config['kdd_model']['emb_dropout'],
            'enc_dropout': config['kdd_model']['enc_dropout'],
            'embedding': config['kdd_model']['embedding'],
            'conv_config': conv_config,
            'num_classes': num_classes
        }

        pprint(model_config)
        model = KDDTransformerEncoderClassification(**model_config)

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)

    return model, model_config


def load_create_imputation_model(config: dict) -> (nn.Module, dict):
    if config['pretrain_model_path']:
        sys.exit("Pretrained model loading is not implemented yet.")
        with open(os.path.join(config['pretrain_model_path'], 'kdd_model_config.json'), 'r') as f:
            model_config = json.load(f)

        model = KDDTransformerEncoderImputation(**model_config)

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)

        model.load_state_dict(
            torch.load(os.path.join(config['pretrain_model_path'], 'kdd_model.pth'), map_location=device))

    else:
        if config['kdd_model']['conv_config']:
            conv_config = config[config['kdd_model']['conv_config']]
        else:
            conv_config = None

        model_config = {
            'feat_dim': config['kdd_model']['feat_dim'],
            'max_len': config['kdd_model']['max_seq_len'],
            'd_model': config['kdd_model']['d_model'],
            'n_heads': config['kdd_model']['n_heads'],
            'n_layers': config['kdd_model']['n_layers'],
            'dim_feedforward': config['kdd_model']['dim_feedforward'],
            'emb_dropout': config['kdd_model']['emb_dropout'],
            'enc_dropout': config['kdd_model']['enc_dropout'],
            'embedding': config['kdd_model']['embedding'],
            'conv_config': conv_config
        }

        pprint(model_config)
        model = KDDTransformerEncoderImputation(**model_config)

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)

    return model, model_config


def load_create_original_classification_model(config: dict, num_classes: int) -> (nn.Module, dict):
    if config['pretrain_model_path']:
        # sys.exit("Pretrained model loading is not implemented yet.")
        with open(os.path.join(config['pretrain_model_path'], 'kdd_model_config.json'), 'r') as f:
            model_config = json.load(f)

        model_config['num_classes'] = num_classes
        model = KDDOriginalTransformerEncoderClassification(**model_config)

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)

        # Load the pre-trained state dict
        pretrained_state_dict = torch.load(os.path.join(config['pretrain_model_path'], 'best_model.pth'), map_location=device)

        # Remove the 'output' layer from the pre-trained state dict
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('output')}

        # Load the modified state dict
        model_state_dict = model.state_dict()
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict, strict=False)

        print("Pretrained model loaded successfully (without output layer).")

    else:
        if config['kdd_original_model']['conv_config']:
            conv_config = config[config['kdd_original_model']['conv_config']]
        else:
            conv_config = None

        model_config = {
            'feat_dim': config['kdd_original_model']['feat_dim'],
            'max_len': config['kdd_original_model']['max_seq_len'],
            'd_model': config['kdd_original_model']['d_model'],
            'n_heads': config['kdd_original_model']['n_heads'],
            'n_layers': config['kdd_original_model']['n_layers'],
            'dim_feedforward': config['kdd_original_model']['dim_feedforward'],
            'emb_dropout': config['kdd_original_model']['emb_dropout'],
            'enc_dropout': config['kdd_original_model']['enc_dropout'],
            'embedding': config['kdd_original_model']['embedding'],
            'conv_config': conv_config,
            'num_classes': num_classes,
            'pre_norm': config['kdd_original_model']['pre_norm']
        }

        pprint(model_config)
        model = KDDOriginalTransformerEncoderClassification(**model_config)

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)

    return model, model_config


def load_create_original_imputation_model(config: dict) -> (nn.Module, dict):
    if config['pretrain_model_path']:
        sys.exit("Pretrained model loading is not implemented yet.")
        with open(os.path.join(config['pretrain_model_path'], 'kdd_model_config.json'), 'r') as f:
            model_config = json.load(f)

        model = KDDOriginalTransformerEncoderImputation(**model_config)

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)

        model.load_state_dict(
            torch.load(os.path.join(config['pretrain_model_path'], 'kdd_model.pth'), map_location=device))

    else:
        if config['kdd_original_model']['conv_config']:
            conv_config = config[config['kdd_original_model']['conv_config']]
        else:
            conv_config = None

        model_config = {
            'feat_dim': config['kdd_original_model']['feat_dim'],
            'max_len': config['kdd_original_model']['max_seq_len'],
            'd_model': config['kdd_original_model']['d_model'],
            'n_heads': config['kdd_original_model']['n_heads'],
            'n_layers': config['kdd_original_model']['n_layers'],
            'dim_feedforward': config['kdd_original_model']['dim_feedforward'],
            'emb_dropout': config['kdd_original_model']['emb_dropout'],
            'enc_dropout': config['kdd_original_model']['enc_dropout'],
            'embedding': config['kdd_original_model']['embedding'],
            'conv_config': conv_config,
            'pre_norm': config['kdd_original_model']['pre_norm']
        }

        pprint(model_config)
        model = KDDOriginalTransformerEncoderImputation(**model_config)

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)

    return model, model_config
