import itertools
import math
import torch
import time
import torchtext
import typing
import numpy as np


class TransformerModelClassifier(torch.nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, context_length: int = 5000, classes: int = 4):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout, context_length)
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = torch.nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = torch.nn.Linear(d_model, classes)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # if in eval mode
        if not self.transformer_encoder.training:# and not self.has_run:
            #self.has_run = True
            #print("src", list(np.array(src.cpu()).tolist()))
            src = self.encoder(src) * math.sqrt(self.d_model)
            #print("src 0", list(src[:, 0, :].cpu().detach().numpy().tolist()))
            src = self.pos_encoder(src)
            #print("src 1", list(src[:, 0, :].cpu().detach().numpy().tolist()))
            output = self.transformer_encoder(src, src_mask)
            print("output 0", list(output[:, 0, :].cpu().detach().numpy().tolist()))
            output = self.decoder(output)
            #print("output 1", list(output[:, 0, :].cpu().detach().numpy().tolist()))
        else:
            src = self.encoder(src) * math.sqrt(self.d_model)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src, src_mask)
            output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def get_model(ntokens: int, emsize: int, nhead: int, d_hid: int, nlayers: int, dropout: float, context_length: int = 5000) -> TransformerModelClassifier:
    return TransformerModelClassifier(ntokens, emsize, nhead, d_hid, nlayers, dropout, context_length)


def get_weights(model):
    weights_1 = model.transformer_encoder.layers[1].weight
    biases_1 = model.transformer_encoder.layers[1].bias
    weights_0 = model.transformer_encoder.layers[0].weight
    biases_0 = model.transformer_encoder.layers[0].bias
    return weights_1, biases_1, weights_0, biases_0


def train_epoch(model: torch.nn.Module, ntokens: int, epoch: int, criterion, optimizer, scheduler, train_data, bptt,
                device) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data[0].size(0) - 1, bptt)):
        data, targets = get_batch(train_data)
        data.to(device)
        targets.to(device)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def get_batch(source: typing.Tuple[torch.Tensor, torch.Tensor], ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    return source[0][:100], source[1][:100]


def evaluate(model: torch.nn.Module, eval_data: typing.Tuple[torch.Tensor, torch.Tensor], device, ntokens: int = 28782, criterion=torch.nn.CrossEntropyLoss(),
             bptt: int = 35) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
            #print('data, loss')
            #print(data, criterion(output_flat, targets).item())
            #print('data, loss')
    return total_loss / (len(eval_data) - 1)


def get_device():
    return 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')


def train(model: torch.nn.Module, train_data, device, name: str = "1", epochs: int = 1, ntokens: int = 28782, lr: int = 5.0) -> torch.nn.Module:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    gamma = (0.03 / lr) ** (1 / epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=gamma)
    bptt = 35

    model.to(device)
    for epoch in range(1, epochs + 1):
        train_epoch(model, ntokens, epoch, criterion, optimizer, scheduler, train_data, bptt, device)
        scheduler.step()
    torch.save(model, f'model_{name}.pt')

    return model
