"""Train a sequence to sequence model on WMT data"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from preprocess import read_file, filterPairs, build_word_index, get_data
import pickle
import numpy as np
import argparse
import logging
import sys
import time
from datetime import datetime
import matplotlib.pyplot as plt
import random


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f'Using device {device}')

# handling seeding stuff for deterministic execution
torch.manual_seed(0)
if str(device) == 'cuda':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

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
            bidirectional=True
        )

    def forward(self, sentence_input, hidden_state, cell_state):
        embedded = self.embedding(sentence_input).view(1, 1, -1)
        out, (hidden_out, cell_out) = self.LSTM(
            embedded, (hidden_state, cell_state))

        return out, (hidden_out, cell_out)

    def init_hidden(self):
        # num layers * num diretions, batch size, hidden size.
        hidden_state = torch.zeros(2, 1, self.hidden_size, device=device)
        cell_state = torch.zeros(2, 1, self.hidden_size, device=device)

        return hidden_state, cell_state


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(
            num_embeddings=self.output_size,
            embedding_dim=self.hidden_size)
        self.LSTM = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, output_size)
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
        hidden_state = torch.zeros(2, 1, self.hidden_size, device=device)
        cell_state = torch.zeros(2, 1, self.hidden_size, device=device)
        return hidden_state, cell_state


def vectorize(sentence, word_to_idx):
    idxes = torch.LongTensor([word_to_idx[word] for word in sentence.split()])
    return idxes


def predict(encoder, decoder, sentences, word_to_idx, idx_to_word):
    hidden_state, cell_state = encoder.init_hidden()
    # encoder_optimizer.zero_grad()
    # decoder_optimizer.zero_grad()
    for k in range(len(sentences)):
        input_sentence = sentences[k][0]
        label_sentence = sentences[k][1]
        idxes_input = vectorize(input_sentence, word_to_idx)
        idxes_label = vectorize(label_sentence, word_to_idx)
        if str(device) == 'cuda':
            idxes_input = idxes_input.to(device)
            idxes_label = idxes_label.to(device)
        input_len = idxes_input.shape[0]
        for i in range(input_len):
            out, (hidden_state, cell_state) = encoder(
                idxes_input[i], hidden_state, cell_state)

        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
        decoder_hidden = hidden_state
        decoder_cell = cell_state
        output_len = idxes_label.shape[0]
        predicted_words = []
        actual_words = []
        for di in range(output_len):
            decoder_output, (decoder_hidden, decoder_cell) = decoder(
                decoder_input, decoder_hidden, decoder_cell)
            # decoder_input = idxes_label[di].view(1, 1)
            actual_idx = idxes_label[di].view(1, 1).item()
            actual_word = idx_to_word[actual_idx]
            actual_words.append(actual_word)
            _, indices = torch.max(decoder_output, 1)
            decoder_input = indices.view(1, 1)
            prediction_idx = indices.item()
            predicted_word = idx_to_word[prediction_idx]
            if predicted_word == EOS_TOKEN:
                break
            predicted_words.append(predicted_word)
        print(predicted_words)
        print(actual_words)
        predicted_sentence = " ".join(predicted_words)
        actual_sentence = " ".join(actual_words)
        print(f'The original sentence: {input_sentence}')
        print(f'Predicted sentence: {predicted_sentence}')
        print(f'Actual sentence: {actual_sentence}')


def serialize_model(model, name):
    time_str = "".join(str(datetime.now()).split())
    path = 'seq2seq_model_{}_{}'.format(name, time_str)
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=str(device)))
    return model


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
    losses = []
    n_epochs = 50
    teacher_forcing_prob = 0.5
    now = time.time()
    for epoch in range(n_epochs):
        # hidden_state, cell_state = encoder.init_hidden()
        for k in range(len(sentences)):
            hidden_state, cell_state = encoder.init_hidden()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            input_sentence = sentences[k][0]
            label_sentence = sentences[k][1]
            idxes_input = vectorize(input_sentence, word_to_idx)
            idxes_label = vectorize(label_sentence, word_to_idx)
            if str(device) == 'cuda':
                idxes_input = idxes_input.to(device)
                idxes_label = idxes_label.to(device)
            input_len = idxes_input.shape[0]
            for i in range(input_len):
                out, (hidden_state, cell_state) = encoder(
                    idxes_input[i], hidden_state, cell_state)

            decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
            decoder_hidden = hidden_state
            decoder_cell = cell_state
            output_len = idxes_label.shape[0]
            loss = 0
            for di in range(output_len):
                decoder_output, (decoder_hidden, decoder_cell) = decoder(
                    decoder_input, decoder_hidden, decoder_cell)
                loss += criterion(decoder_output, idxes_label[di].view(1))
                use_teacher_forcing = random.random() < 0.5
                if use_teacher_forcing:
                    decoder_input = idxes_label[di].view(1, 1)
                else:
                    # use its own output as the input
                    _, indices = torch.max(decoder_output, 1)
                    decoder_input = indices.view(1, 1)
            loss.backward()
            # gradient clipping, make max gradinent have norm 1.
            # clip_grad_norm_(encoder.parameters(), 1)
            # clip_grad_norm_(decoder.parameters(), 1)
            encoder_optimizer.step()
            decoder_optimizer.step()
            if k % 20 == 0:
                logger.info(f'Current loss: {loss.item()/output_len}, iteration {k} out of {len(sentences)}, epoch:{epoch}')
                losses.append(loss.item() / output_len)
    logger.info(f'Took {time.time()-now} seconds to train')
    plt.plot(range(len(losses)), losses)
    # plt.show()
    plt.savefig('foo2.png')
    # import pdb; pdb.set_trace()


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
    parser.add_argument(
        '--small',
        action='store_true',
        default=False,
        help='Set this is you just want to overfit a small dataset (will be ignored if sentences_data is passed in.')
    parser.add_argument(
        '--save',
        action='store_true',
        default=False,
        help='Pass in if you want to save the model')
    parser.add_argument(
        '--encoder',
        type=str,
        default=None,
        help='path to encoder model if you want to load it, specify this with decoder as well.')
    parser.add_argument(
        '--decoder',
        type=str,
        default=None,
        help='path to decoder model if you want to load it, specify this with encoder as well.')
    args = parser.parse_args()
    short_sentences, idx_to_word, word_to_idx = get_data(args)
    num_words = len(idx_to_word)
    # assert num_words == max(idx_to_word.keys())
    # initialize an encoder with hidden dim of 500
    encoder = Encoder(input_size=num_words, hidden_size=500)
    encoder = encoder.to(device)

    decoder = Decoder(hidden_size=500, output_size=num_words)
    decoder = decoder.to(device)
    if not (args.encoder and args.decoder):
        train_model(
            encoder,
            decoder,
            short_sentences,
            word_to_idx,
            idx_to_word)
    else:
        logger.info(
            'Loading model from paths {} and {}'.format(
                args.encoder, args.decoder))
        encoder = load_model(encoder, args.encoder)
        logger.info('Loaded encoder.')
        decoder = load_model(decoder, args.decoder)
        logger.info('Loaded decoder.')
    if args.save:
        logger.info('Saving model.')
        serialize_model(encoder, 'encoder')
        serialize_model(decoder, 'decoder')
    predict(encoder, decoder, short_sentences, word_to_idx, idx_to_word)
