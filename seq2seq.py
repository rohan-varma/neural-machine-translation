"""Train a sequence to sequence model on WMT data"""
import torch
from torch import nn
from preprocess import normalize_string, filterPairs

# tokens to mark the start and end of the sentence
SOS_TOKEN = 0
EOS_TOKEN = 1
idx_to_word = {0: SOS_TOKEN, 1: EOS_TOKEN}
word_to_idx = dict(zip(idx_to_word.values(), idx_to_word.keys()))
print(idx_to_word)
print(word_to_idx)
next_idx = 2


class seq2seq(nn.Module):
	def __init__(self):
		pass





def read_file(data_file):
	lines = open(data_file, encoding='utf-8').read().strip().split('\n')
	pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
	print(f'size of dataset: {len(pairs)}')
	print(pairs[len(pairs)//2])
	return pairs

def build_word_index(sentences):
	global next_idx
	for pair in sentences:
		for x in pair:
			words = x.split()
			for word in words:
				if word not in word_to_idx:
					# first, add it to idx_to_word
					idx_to_word[next_idx] = word
					# maintain this dinctionary so we dont double add words
					word_to_idx[word] = next_idx
					next_idx+=1 # increment so we don't clash indices



if __name__ == '__main__':
	data_file = 'data/eng-fra.txt'
	sentence_pairs = read_file(data_file)
	print('filtering pairs...')
	short_sentences = filterPairs(sentence_pairs)
	print(f'trimmed dataset to {len(short_sentences)}')
	print(short_sentences[len(short_sentences)//2])
	build_word_index(short_sentences)
	import pdb; pdb.set_trace()
