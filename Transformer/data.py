import itertools
import math
import torch
import time
import torchtext
import typing
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence
import torchtext
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def get_classification_set(device, hetero_split=False, batch_size=20):
    # from: https://github.com/dbetm/my-ai-history/blob/1036aae55a11ed7bf691a2f15a65eefe1b0a077a/courses/Coursera/Intro%20to%20ML%20-%20Duke%20University/NLP/agnews.py

    train_iter, test_iter = torchtext.datasets.AG_NEWS(
        split=('train', 'test')
    )

    tokenizer = get_tokenizer(tokenizer='basic_english', language='en')

    def yield_tokens(data_iter):
        for label, text in data_iter:
            yield tokenizer(text)

    # The vocabulary block converts a list of tokens into integers.
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
    # unk = unknown (default)
    vocab.set_default_index(vocab['<unk>'])

    # example_words = ['cat', 'dog', 'chicken']
    # print(example_words, vocab(example_words))

    # Create data loaders
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')

    def collate_batch(batch):
        labels = torch.tensor([label_pipeline(example[0]) for example in batch])
        sentences = [torch.tensor(text_pipeline(example[1])) for example in batch]
        data = pad_sequence(sentences).clone().detach()

        return [data, labels]

    train_iter, test_iter = torchtext.datasets.AG_NEWS(
        split=('train', 'test')
    )

    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)

    BATCH_SIZE = 128

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch
    )

    train_iter, test_iter = torchtext.datasets.AG_NEWS(
        split=('train', 'test')
    )

    train_data = data_process(train_iter, vocab, tokenizer)
    test_data = data_process(test_iter, vocab, tokenizer)

    train_data[0].to(device)
    train_data[1].to(device)
    test_data[0].to(device)
    test_data[1].to(device)

    return train_data, test_data


def get_dataset_split(device, type='wiki', hetero_split=False, batch_size=20):
    train_iter_wiki = torchtext.datasets.WikiText2(split='train')
    train_iter_penn = torchtext.datasets.WikiText2(split='train')
    train_iter = itertools.chain(train_iter_wiki, train_iter_penn)
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    train_iter_penn, val_iter_penn, test_iter_penn = torchtext.datasets.PennTreebank()
    train_data_penn = data_process(train_iter_penn, vocab, tokenizer)
    val_data_penn = data_process(val_iter_penn, vocab, tokenizer)
    test_data_penn = data_process(test_iter_penn, vocab, tokenizer)

    train_iter_wiki, val_iter_wiki, test_iter_wiki = torchtext.datasets.WikiText2()
    train_data_wiki = data_process(train_iter_wiki, vocab, tokenizer)
    val_data_wiki = data_process(val_iter_wiki, vocab, tokenizer)
    test_data_wiki = data_process(test_iter_wiki, vocab, tokenizer)
    if not type == 'wiki':
        train_data_wiki = train_data_wiki[:train_data_penn.shape[0]]
        val_data_wiki = train_data_wiki[:val_data_penn.shape[0]]
        test_data_wiki = train_data_wiki[:test_data_penn.shape[0]]

    eval_batch_size = 2
    train_data_wiki = batchify(train_data_wiki, batch_size, device)  # shape [seq_len, batch_size]
    val_data_wiki = batchify(val_data_wiki, eval_batch_size, device)
    test_data_wiki = batchify(test_data_wiki, eval_batch_size, device)

    train_data_penn = batchify(train_data_penn, batch_size, device)  # shape [seq_len, batch_size]
    val_data_penn = batchify(val_data_penn, eval_batch_size, device)
    test_data_penn = batchify(test_data_penn, eval_batch_size, device)

    if type == 'wiki':
        data_1, data_2, _ = train_data_wiki.split(train_data_wiki.size(0) // 2, dim=0)
        val_data = val_data_wiki
        test_data = test_data_wiki
    elif type == 'penn':
        data_1, data_2, _ = train_data_penn.split(train_data_penn.size(0) // 2, dim=0)
        val_data = val_data_penn
        test_data = test_data_penn
    elif type == 'hetero':
        data_1 = train_data_wiki
        data_2 = train_data_penn
        val_data = torch.cat((val_data_wiki[:210], val_data_penn[:210]), dim=1)
        test_data = torch.cat((test_data_wiki, test_data_penn), dim=1)
    else:
        raise ValueError('Invalid type')

    if hetero_split:
        train_data = torch.cat((data_1, data_2), dim=1)
        datas = train_data.split(train_data.size(0) // 10000, dim=0)
        averages = [np.array(data.cpu()).mean() for data in datas]
        # get the indices of the 50% of the data with the highest average and the 50% with the lowest average
        indices = np.argsort(averages)
        low_datas = [datas[i] for i in indices[:len(indices) // 2]]
        high_datas = [datas[i] for i in indices[len(indices) // 2:]]
        data_1 = torch.cat(low_datas, dim=0)
        data_2 = torch.cat(high_datas, dim=0)

    return data_1, data_2, val_data, test_data


def get_original_dataset_split(device):
    train_iter = torchtext.datasets.WikiText2(split='train')
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = torchtext.datasets.WikiText2()
    train_data = data_process(train_iter, vocab, tokenizer)
    val_data = data_process(val_iter, vocab, tokenizer)
    test_data = data_process(test_iter, vocab, tokenizer)

    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_data, batch_size, device)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, eval_batch_size, device)
    test_data = batchify(test_data, eval_batch_size, device)

    data_1, data_2, _ = train_data.split(train_data.size(0) // 2, dim=0)
    return data_1, data_2, val_data, test_data


def data_process(raw_text_iter: torch.utils.data.dataset.IterableDataset, vocab, tokenizer) -> torch.Tensor:
    """Converts raw text into a flat Tensor."""
    data = [(label, torch.tensor(vocab(tokenizer(item)), dtype=torch.long)) for label, item in raw_text_iter]
    data_short = [(label, item) for label, item in data if (35 > item.numel() > 0)]
    items_short = [item for _, item in data_short]
    labels_short = [label for label, _ in data_short]
    data_pad = [torch.nn.functional.pad(item, (0, 35 - item.numel()), 'constant', 0) for item in items_short]
    data_pad_tensor = torch.stack(data_pad, dim=0)
    label_tensor = torch.tensor(labels_short, dtype=torch.long)
    return data_pad_tensor, label_tensor


def batchify(data: torch.Tensor, bsz: int, device) -> torch.Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)
