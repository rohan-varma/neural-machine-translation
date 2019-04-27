"""Train a sequence to sequence model on WMT data"""
import torch
from torch import nn
from preprocess import read_file, filterPairs, build_word_index, get_data
import pickle
import argparse
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()

# tokens to mark the start and end of the sentence
SOS_TOKEN = 0
EOS_TOKEN = 1


class seq2seq(nn.Module):
	def __init__(self):
		pass



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Attention model training')
	parser.add_argument('--sentences_data', type=str, help='file containing list of sentences and index generated, outputted by this script')
	parser.add_argument('--data', type=str, default='data/eng-fra.txt', help='dataset of english to french translation')
	parser.add_argument('--save_processed', default='data/eng-fra-processed.txt', help='whether to save processed list and index of dataset')
	args = parser.parse_args()
	short_sentences, idx_to_word, word_to_idx = get_data(args)