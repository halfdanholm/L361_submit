import copy
import merge
import transformer
import torch
import sys
import argparse
import data

import copy
import merge
import transformer
import torch
import sys
import argparse
import data
from collections import OrderedDict
from torch import tensor


def main():
    emsize = 2  # embedding dimension
    d_hid = 2  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 1  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    ntokens = 2  # size of vocabulary
    context_length = 3

    model_orig = transformer.get_model(ntokens, emsize, nhead, d_hid, nlayers, dropout, context_length)
    model_permuted = transformer.get_model(ntokens, emsize, nhead, d_hid, nlayers, dropout, context_length)

    queries = tensor([
        [.29, .43],
        [.41, .31],
    ])
    queries_permuted = tensor([
        [.41, .31],
        [.29, .43],
    ])

    keys = tensor([
        [.2, .3],
        [.5, .7],
    ])
    keys_permuted = tensor([
        [.5, .7],
        [.2, .3],
    ])

    vals = tensor([
        [.2, .31],
        [.41, .7],
    ])
    vals_permuted = tensor([
        [.41, .7],
        [.2, .31],
    ])

    queries_bias = tensor([0, 0])
    keys_bias = tensor([0, 0])
    vals_bias = tensor([0, 0])

    out_proj = tensor([
        [1, 0],
        [0, 1],
    ])
    out_proj_permuted = tensor([
        [0, 1],
        [1, 0],
    ])
    out_proj_bias = tensor([0, 0])

    pos_encoder = tensor([
        [[0, 0]],
        [[0, 0]],
        [[0, 0]],
    ])
    encoder = tensor([
        [.11, .23],
        [.17, .13],
    ])
    decoder = tensor([
        [1, 0],
        [0, 1],
    ])
    decoder_bias = tensor([0, 0])

    lin_1 = tensor([
        [1, 0],
        [0, 1],
    ])
    lin_1_bias = tensor([0, 0])
    lin_2 = tensor([
        [1, 0],
        [0, 1],
    ])
    lin_2_bias = tensor([0, 0])
    norm_1 = tensor([1, 1])
    norm_1_bias = tensor([0, 0])
    norm_2 = tensor([1, 1])
    norm_2_bias = tensor([0, 0])

    state_dict = OrderedDict([
        ('pos_encoder.pe', pos_encoder),
        ('transformer_encoder.layers.0.self_attn.in_proj_weight', torch.cat((queries, keys, vals), dim=0)),
        ('transformer_encoder.layers.0.self_attn.in_proj_bias', torch.cat((queries_bias, keys_bias, vals_bias), dim=0)),
        ('transformer_encoder.layers.0.self_attn.out_proj.weight', out_proj),
        ('transformer_encoder.layers.0.self_attn.out_proj.bias', out_proj_bias),
        ('transformer_encoder.layers.0.linear1.weight', lin_1),
        ('transformer_encoder.layers.0.linear1.bias', lin_1_bias),
        ('transformer_encoder.layers.0.linear2.weight', lin_2),
        ('transformer_encoder.layers.0.linear2.bias', lin_2_bias),
        ('transformer_encoder.layers.0.norm1.weight', norm_1),
        ('transformer_encoder.layers.0.norm1.bias', norm_1_bias),
        ('transformer_encoder.layers.0.norm2.weight', norm_2),
        ('transformer_encoder.layers.0.norm2.bias', norm_2_bias),
        ('encoder.weight', encoder),
        ('decoder.weight', decoder),
        ('decoder.bias', decoder_bias)
    ])

    """state_dict['pos_encoder.pe'] = model.state_dict()['pos_encoder.pe']
    state_dict['encoder.weight'] = model.state_dict()['encoder.weight']
    state_dict['decoder.weight'] = model.state_dict()['decoder.weight']
    state_dict['decoder.bias'] = model.state_dict()['decoder.bias']"""

    torch.set_printoptions(threshold=10000)

    model_orig.load_state_dict(state_dict)
    torch.save(model_orig, 'orig.pt')

    state_dict_permuted = copy.deepcopy(state_dict)
    state_dict_permuted['transformer_encoder.layers.0.self_attn.in_proj_weight'] = torch.cat((queries_permuted, keys_permuted, vals_permuted), dim=0)
    state_dict_permuted['transformer_encoder.layers.0.self_attn.out_proj.weight'] = out_proj_permuted
    model_permuted.load_state_dict(state_dict_permuted)
    torch.save(model_permuted, 'permuted.pt')

    """print(state_dict)
    for key, val in state_dict.items():
        if key not in ['pos_encoder.pe', 'encoder.weight', 'decoder.weight', 'decoder.bias']:
            print(key)
            print(val.shape)
            print(list(val))"""

    device = 'cpu'#transformer.get_device()
    simple_data = torch.tensor([[0], [1], [0], [1]])
    simple_data.to(device)
    """data_1_orig, data_2_orig, val_data_orig, _ = data.get_original_dataset_splits(device)
    simple_data = val_data_orig[:, :2] % 2
    simple_data.to(device)"""

    model_orig.to(device)
    model_permuted.to(device)

    out_orig = get_output(model_orig, simple_data, device)
    print(out_orig)

    out_permuted = get_output(model_permuted, simple_data, device)
    print(out_permuted)


def get_output(model, eval_data, device):
    model.eval()
    bptt = 3
    src_mask = transformer.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = transformer.get_batch(eval_data, i, bptt)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            return model(data, src_mask)

if __name__ == '__main__':
    main()
