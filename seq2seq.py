"""Train a sequence to sequence model on WMT data"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from preprocess import read_file, filterPairs, build_word_index, get_data
import pickle
import numpy as np
import argparse
import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Using device {device}')

# tokens to mark the start and end of the sentence
SOS_TOKEN = 0
EOS_TOKEN = 1


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # we want an embedding matrix of num_total_words * dim_for_each_word
        self.embedding = nn.Embedding(
            num_embeddings=self.input_size,
            embedding_dim=hidden_size)
        self.LSTM = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )

    def forward(self, sentence_input, hidden_state, cell_state):
        embedded = self.embedding(sentence_input).view(1, 1, -1)
        out, (hidden_out, cell_out) = self.LSTM(
            embedded, (hidden_state, cell_state))

        return out, (hidden_out, cell_out)

    def init_hidden(self):
        # num layers * num diretions, batch size, hidden size.
        hidden_state = torch.zeros(1, 1, self.hidden_size, device=device)
        cell_state = torch.zeros(1, 1, self.hidden_size, device=device)

        return hidden_state, cell_state


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(
            num_embeddings=self.output_size,
            embedding_dim=self.hidden_size)
        self.LSTM = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, word_input, hidden_state, cell_state):
        out = self.embedding(word_input).view(1, 1, -1)
        out = F.relu(out)
        out, (hidden_state, cell_state) = self.LSTM(
            out, (hidden_state, cell_state))
        out = self.softmax(self.linear(out[0]))
        return out, (hidden_state, cell_state)

    def init_hidden(self):
        # num layers * num diretions, batch size, hidden size.
        hidden_state = torch.zeros(1, 1, self.hidden_size, device=device)
        cell_state = torch.zeros(1, 1, self.hidden_size, device=device)
        return hidden_state, cell_state


def vectorize(sentence, word_to_idx):
    idxes = torch.LongTensor([word_to_idx[word] for word in sentence.split()])
    return idxes


def train_model(encoder, decoder, sentences, word_to_idx, idx_to_word):
    encoder_optimizer = optim.Adam(
        encoder.parameters(),
        lr=1e-4 / 3,
        weight_decay=.00001)
    decoder_optimizer = optim.Adam(
        decoder.parameters(),
        lr=1e-4 / 3,
        weight_decay=.00001)
    criterion = nn.CrossEntropyLoss()
    # hidden_state, cell_state = encoder.init_hidden()

    for k in range(500):
        hidden_state, cell_state = encoder.init_hidden()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        input_sentence = sentences[k][0]
        label_sentence = sentences[k][1]
        idxes = vectorize(input_sentence, word_to_idx)
        idxes_label = vectorize(label_sentence, word_to_idx)
        input_len = idxes.shape[0]
        for i in range(input_len):
            out, (hidden_state, cell_state) = encoder(
                idxes[i], hidden_state, cell_state)

        logger.debug(f'Encoder output shape: {out.shape}, hidden state shape: {hidden_state.shape}')

        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
        decoder_hidden = hidden_state
        decoder_cell = cell_state
        output_len = idxes_label.shape[0]
        loss = 0
        for di in range(output_len):
            decoder_output, (decoder_hidden, decoder_cell) = decoder(
                decoder_input, decoder_hidden, decoder_cell)
            loss += criterion(decoder_output, idxes_label[di].view(1))
            decoder_input = idxes_label[di].view(1, 1)
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        logger.info(f'Current loss: {loss.item()/output_len}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attention model training')
    parser.add_argument(
        '--sentences_data',
        type=str,
        help='file containing list of sentences and index generated, outputted by this script')
    parser.add_argument(
        '--data',
        type=str,
        default='data/eng-fra.txt',
        help='dataset of english to french translation')
    parser.add_argument(
        '--save_processed',
        default='data/eng-fra-processed.txt',
        help='whether to save processed list and index of dataset')
    parser.add_argument(
        '--no_trim',
        action='store_true',
        default=False,
        help='Specify this if you want to train on the full dataset instead of the trimmed version.')
    args = parser.parse_args()
    short_sentences, idx_to_word, word_to_idx = get_data(args)
    num_words = len(idx_to_word) - 1
    assert num_words == max(idx_to_word.keys())
    # initialize an encoder with hidden dim of 100
    encoder = Encoder(input_size=num_words, hidden_size=100)
    encoder = encoder.to(device)

    decoder = Decoder(hidden_size=100, output_size=num_words)
    train_model(encoder, decoder, short_sentences, word_to_idx, idx_to_word)
