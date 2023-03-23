import copy
import merge
import transformer_classification
import torch
import sys
import argparse
import data


def main():
    emsize = 10  # embedding dimension
    d_hid = 6  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 1  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 5  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    ntokens = 28782  # size of vocabulary

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoints_folder", type=str, default="skip", help="Checkpoint Folder")
    parser.add_argument("--data_type", type=str, default="hetero")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--diff_init", action="store_true")
    parser.add_argument("-r", "--random_permutation", action="store_true")
    parser.add_argument("--lr", type=int, default=5.0)
    args = parser.parse_args()

    device = transformer_classification.get_device()
    #data_1_orig, data_2_orig, val_data_orig, _ = data.get_original_dataset_split(device)
    #data_1, data_2, val_data, _ = data.get_dataset_split(device, type=args.data_type, batch_size=args.batch_size)
    #val_data = val_data[:5, :1]
    data_c_1, data_c_val = data.get_classification_set(device, hetero_split=False, batch_size=args.batch_size)

    if args.checkpoints_folder != 'skip':
        print('Loading checkpoints...')
        model_1_trained = torch.load(f'{sys.argv[2]}/model_1.pt', map_location=device)
        model_2_trained = torch.load(f'{sys.argv[2]}/model_2.pt', map_location=device)
        model_1_trained.to(device)
        model_2_trained.to(device)
    else:
        model_1 = transformer_classification.get_model(ntokens, emsize, nhead, d_hid, nlayers, dropout)
        model_1.to(device)
        if args.diff_init:
            model_2 = transformer_classification.get_model(ntokens, emsize, nhead, d_hid, nlayers, dropout)
            model_2.to(device)
        else:
            model_2 = copy.deepcopy(model_1)
            model_2.to(device)
        print('Training model 1...')
        model_1_trained = transformer_classification.train(model_1, data_c_1, device, name='1', epochs=args.epochs, lr=args.lr)
        print('Training model 2...')
        model_2_trained = transformer_classification.train(model_2, data_c_1, device, name='2', epochs=args.epochs, lr=args.lr)

    print(model_1_trained)
    print('Got models')

    model_permuted = merge.permute_model(device, model_1_trained, model_2_trained, random_permuation=args.random_permutation)
    model_permuted.to(device)

    model_merged = merge.average_model(model_1_trained, model_permuted)
    model_merged.to(device)

    model_av = merge.average_model(model_1_trained, model_2_trained)
    model_av.to(device)

    loss_2 = transformer_classification.evaluate(model_2_trained, data_c_val, device, bptt=5)
    print(f'Loss 2: {loss_2}')
    loss_permuted = transformer_classification.evaluate(model_permuted, data_c_val, device, bptt=5)
    print(f'Loss permuted: {loss_permuted}')
    loss_merged = transformer_classification.evaluate(model_merged, data_c_val, device)
    print(f'Loss merged: {loss_merged}')
    loss_av = transformer_classification.evaluate(model_av, data_c_val, device)
    print(f'Loss average: {loss_av}')
    loss_1 = transformer_classification.evaluate(model_1_trained, data_c_val, device)
    print(f'Loss 1: {loss_1}')


if __name__ == '__main__':
    main()
