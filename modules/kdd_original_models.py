import sys
from math import floor
from torch.nn import functional as F
from torch.nn import TransformerEncoderLayer

from modules.modified_transformer import KDDOriginalLearnablePositionalEncoding, TransformerBatchNormEncoderLayer
from einops.layers.torch import Rearrange
from torch import nn


class LinearLengthAdjust(nn.Module):
    def __init__(self, input_length, target_length, feat_dim):
        super().__init__()
        self.linear = nn.Linear(input_length * feat_dim, target_length * feat_dim)
        self.input_length = input_length
        self.target_length = target_length
        self.feat_dim = feat_dim

    def forward(self, x):
        batch_size = x.size(0)
        x = x.contiguous().view(batch_size, -1)  # Ensure the tensor is contiguous and then flatten
        x = self.linear(x)
        return x.view(batch_size, self.target_length, self.feat_dim)


class KDDOriginalTransformerEncoderImputation(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, n_layers, dim_feedforward, emb_dropout=0.1, enc_dropout=0.1, embedding='convolution', conv_config=None, pre_norm=False):
        super(KDDOriginalTransformerEncoderImputation, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        if embedding == "linear":
            self.proj_inp = nn.Linear(feat_dim, d_model)
            print(f"Linear embedding: {self.max_len} sequence length.")
        elif embedding == "convolution":
            assert conv_config is not None, "Embedding is chosen as Conv, but conv_config is empty."
            self.proj_inp = nn.Sequential(
                Rearrange('b l d -> b d l'),  # Rearrange input shape to [batch_size, feat_dim, seq_length]
                nn.Conv1d(feat_dim, d_model, kernel_size=conv_config['kernel_size'], stride=conv_config['stride'],
                          padding=conv_config['padding'], dilation=conv_config['dilation']),
                Rearrange('b d l -> b l d')  # Rearrange output shape to [batch_size, seq_length, d_model]
            )
            proj_conv_seq_len = int(floor((self.max_len + 2 * conv_config['padding'] - conv_config['dilation'] * (
                        conv_config['kernel_size'] - 1) - 1) / conv_config['stride'] + 1))
            self.max_len = proj_conv_seq_len
            print(f"Convolutional embedding: {self.max_len} sequence length.")
        else:
            sys.exit("Embedding should be either 'linear' or 'convolution'.")

        self.pos_embedding = KDDOriginalLearnablePositionalEncoding(self.max_len, d_model)

        self.emb_dropout = nn.Dropout(emb_dropout)

        encoder_layer = TransformerBatchNormEncoderLayer(self.d_model, self.n_heads, dim_feedforward, enc_dropout, pre_norm=pre_norm)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers, enable_nested_tensor=False)

        self.act = F.gelu

        self.enc_dropout = nn.Dropout(enc_dropout)

        if embedding == "linear":
            self.output = nn.Linear(d_model, feat_dim)
        else:  # convolution
            recon_conv_seq_len = (self.max_len - 1) * conv_config['stride'] - 2 * conv_config['padding'] + conv_config['dilation'] * (conv_config['kernel_size'] - 1) + 0 + 1
            self.output = nn.Sequential(
                Rearrange('b l d -> b d l'),
                nn.ConvTranspose1d(d_model, feat_dim, kernel_size=conv_config['kernel_size'],
                                   stride=conv_config['stride'],
                                   padding=conv_config['padding'], dilation=conv_config['dilation']),
                Rearrange('b d l -> b l d'),
                LinearLengthAdjust(recon_conv_seq_len, max_len, feat_dim)
            )
            # recon_conv_seq_len = (self.max_len - 1) * conv_config['stride'] - 2 * conv_config['padding'] + conv_config[
            #     'dilation'] * (conv_config['kernel_size'] - 1) + 0 + 1
            # assert recon_conv_seq_len == max_len, f"Reconstructed sequence length {recon_conv_seq_len} does not match the original sequence length {max_len}. Please choose the Convolutional configuration properly."

    def forward(self, x):
        x = self.proj_inp(x) * (self.d_model ** 0.5)
        x = self.pos_embedding(x)
        x = self.emb_dropout(x)
        x = self.transformer_encoder(x)
        x = self.act(x)
        x = self.enc_dropout(x)
        x = self.output(x)
        return x


class KDDOriginalTransformerEncoderClassification(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, n_layers, dim_feedforward, emb_dropout=0.1, enc_dropout=0.1, embedding='convolution', conv_config=None, num_classes=0, pre_norm=False):
        super(KDDOriginalTransformerEncoderClassification, self).__init__()

        assert num_classes != 0, "Number of classes should be provided for finetuning."
        # print(f"Number of classes: {num_classes}")

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        if embedding == "linear":
            self.proj_inp = nn.Linear(feat_dim, d_model)
            print(f"Linear embedding: {self.max_len} sequence length.")
        elif embedding == "convolution":
            assert conv_config is not None, "Embedding is chosen as Conv, but conv_config is empty."
            self.proj_inp = nn.Sequential(
                Rearrange('b l d -> b d l'),  # Rearrange input shape to [batch_size, feat_dim, seq_length]
                nn.Conv1d(feat_dim, d_model, kernel_size=conv_config['kernel_size'], stride=conv_config['stride'],
                          padding=conv_config['padding'], dilation=conv_config['dilation']),
                Rearrange('b d l -> b l d')  # Rearrange output shape to [batch_size, seq_length, d_model]
            )
            proj_conv_seq_len = int(floor((self.max_len + 2 * conv_config['padding'] - conv_config['dilation'] * (
                        conv_config['kernel_size'] - 1) - 1) / conv_config['stride'] + 1))
            self.max_len = proj_conv_seq_len
            print(f"Convolutional embedding: {self.max_len} sequence length.")
        else:
            sys.exit("Embedding should be either 'linear' or 'convolution'.")

        self.pos_embedding = KDDOriginalLearnablePositionalEncoding(self.max_len, d_model)

        self.emb_dropout = nn.Dropout(emb_dropout)

        encoder_layer = TransformerBatchNormEncoderLayer(self.d_model, self.n_heads, dim_feedforward, enc_dropout, pre_norm=pre_norm)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers, enable_nested_tensor=False)

        self.act = F.gelu

        self.enc_dropout = nn.Dropout(enc_dropout)

        self.output = nn.Sequential(
            Rearrange('b l d -> b (l d)'),
            nn.Linear(self.max_len * d_model, num_classes)
        )

    def forward(self, x):
        x = self.proj_inp(x) * (self.d_model ** 0.5)
        x = self.pos_embedding(x)
        x = self.emb_dropout(x)
        x = self.transformer_encoder(x)
        x = self.act(x)
        x = self.enc_dropout(x)
        x = self.output(x)
        return x
