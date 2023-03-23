import transformer
import torch
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoints_folder", type=str, default="skip", help="Checkpoint Folder")
    args = parser.parse_args()

    device = 'cpu'#transformer.get_device()
    simple_data = torch.tensor([[0], [1], [2]])
    simple_data.to(device)
    """data_1_orig, data_2_orig, val_data_orig, _ = data.get_original_dataset_splits(device)
    simple_data = val_data_orig[:, :2] % 2
    simple_data.to(device)"""

    print('Loading checkpoints...')
    model_orig = torch.load(f'{sys.argv[2]}/orig.pt', map_location=device)
    model_permuted = torch.load(f'{sys.argv[2]}/permuted.pt', map_location=device)
    model_orig.to(device)
    model_permuted.to(device)

    out_orig = get_output(model_orig, simple_data, device)
    out_permuted = get_output(model_permuted, simple_data, device)

    print(out_orig)
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

    """model.eval()
    bptt = 2
    src_mask = transformer.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        seq_len = data.size(0)
        if seq_len != bptt:
            src_mask = src_mask[:seq_len, :seq_len]
        return model(data, src_mask)"""


if __name__ == '__main__':
    main()
