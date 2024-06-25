import json
import os
import sys
from pprint import pprint

import torch
from torch import nn

from modules.kdd_models import KDDTransformerEncoderClassification, KDDTransformerEncoderImputation


def load_create_classification_model(config: dict, num_classes: int) -> (nn.Module, dict):
    if config['pretrain_model_path']:
        sys.exit("Pretrained model loading is not implemented yet.")
        with open(os.path.join(config['pretrain_model_path'], 'kdd_model_config.json'), 'r') as f:
            model_config = json.load(f)

        model = KDDTransformerEncoderClassification(**model_config)

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
