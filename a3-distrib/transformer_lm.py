# models.py
import random

import numpy as np
import torch
from torch import nn


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class CharacterTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=2, max_seq_len=1000):
        super(CharacterTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions=max_seq_len, batched=True)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x, src_mask):
        token_embedding = self.embedding(x)
        x = self.positional_encoding(token_embedding)
        x = self.transformer(x, src_mask)
        logits = self.fc(x)
        return logits

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask


class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_size, vocab_index, d_model=256, nhead=8, num_layers=2, max_seq_len=1000):
        self.model = CharacterTransformer(vocab_size, d_model, nhead, num_layers, max_seq_len)
        self.vocab_size = vocab_size
        self.vocab_index = vocab_index

    def get_next_char_log_probs(self, context):
        self.model.eval()
        with torch.no_grad():
            context_indices = [self.vocab_index.index_of(c) for c in context]
            context_tensor = torch.tensor(context_indices).unsqueeze(0).long()
            src_mask = self.model.generate_square_subsequent_mask(len(context)).to(context_tensor.device)
            logits = self.model(context_tensor, src_mask)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            return log_probs.squeeze(0).cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        self.model.eval()
        with torch.no_grad():
            context_indices = [self.vocab_index.index_of(c) for c in context]
            next_chars_indices = [self.vocab_index.index_of(c) for c in next_chars]
            context_tensor = torch.tensor(context_indices + next_chars_indices).unsqueeze(0).long()
            src_mask = self.model.generate_square_subsequent_mask(len(context_indices) + len(next_chars_indices))
            src_mask = src_mask.to(context_tensor.device)
            logits = self.model(context_tensor, src_mask)
            log_probs = torch.log_softmax(logits, dim=-1)
            sequence_log_prob = sum(
                log_probs[0, i, next_chars_indices[i]].item()
                for i in range(len(next_chars_indices))
            )
            return sequence_log_prob


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    learning_rate = 0.08
    epochs = 10
    chunk_size = 800
    vocab_size = len(vocab_index)
    model = NeuralLanguageModel(vocab_size, vocab_index)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Prepare data for training

    train_indices = np.array([vocab_index.index_of(ci) for ci in train_text])
    dev_indices = np.array([vocab_index.index_of(ci) for ci in dev_text])

    # train_tensor = torch.tensor(train_indices)
    # dev_tensor = torch.tensor(dev_indices)

    # num_chunks = len(train_indices) // chunk_size

    train_chunks = [train_indices[i:i + chunk_size] for i in range(0, len(train_indices), chunk_size)]
    random.shuffle(train_chunks)


    model.model.train()
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        for chunk in train_chunks:
            optimizer.zero_grad()
            chunk_tensor = torch.tensor(chunk).long()
            src_mask = model.model.generate_square_subsequent_mask(len(chunk))
            # print(chunk_tensor.unsqueeze(0).shape)
            logits = model.model(chunk_tensor.unsqueeze(0), src_mask)
            loss = criterion(logits.view(-1, vocab_size), chunk_tensor)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_chunks)}")

    return model
