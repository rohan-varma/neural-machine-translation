# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
import unicodedata
import re
import pickle
import logging
import sys
import random
random.seed(0)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


MAX_LENGTH = 10 # change this to allow for longer or shorter sentences to be translated.
# tokens to mark the start and end of the sentence
SOS_TOKEN = 0
EOS_TOKEN = 1
idx_to_word = {0: SOS_TOKEN, 1: EOS_TOKEN}
word_to_idx = dict(zip(idx_to_word.values(), idx_to_word.keys()))
next_idx = 2


def build_word_index(sentences):
    global next_idx
    for pair in sentences:
        for sentence in pair:
            words = sentence.split()
            for word in words:
                if word not in word_to_idx:
                    # first, add it to idx_to_word
                    idx_to_word[next_idx] = word
                    # maintain this dinctionary so we dont double add words
                    word_to_idx[word] = next_idx
                    next_idx += 1  # increment so we don't clash indices


def get_data(args):
    global idx_to_word
    global word_to_idx
    if not args.sentences_data:
        data_file = args.data
        logger.info(f'Reading data from file {args.data}')
        sentence_pairs = read_file(data_file)
        if args.no_trim:
            logger.info('Not filtering dataset, returning full.')
            short_sentences = sentence_pairs
        else:
            logger.info('Filtering dataset...')
            short_sentences = filterPairs(sentence_pairs)
            if args.small:
                logger.info('Cutting dataset size down to 100.')
                short_sentences = short_sentences[:100]
        logger.info(f'trimmed dataset to {len(short_sentences)}')
        random.shuffle(short_sentences)
        build_word_index(short_sentences)
        logger.info('Build word index.')
        if args.save_processed:
            logger.info(f'Dumping data to {args.save_processed}')
            handle = open(args.save_processed, 'wb')
            pickle.dump(short_sentences, handle)
            pickle.dump(idx_to_word, handle)
            pickle.dump(word_to_idx, handle)
            handle.close()
    else:
        # try to read from that file
        try:
            logger.info(f'Reading from {args.sentences_data}')
            handle = open(args.sentences_data, 'rb')
            short_sentences = pickle.load(handle)
            idx_to_word = pickle.load(handle)
            word_to_idx = pickle.load(handle)
        except ValueError:
            logger.error(f'Could not read processed data from {args.sentences_data}, make sure that this file was generated by this script')
        logger.info(
            'Read preprocessed data, got size {}.'.format(
                len(short_sentences)))
    return short_sentences, idx_to_word, word_to_idx


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[0].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def read_file(data_file):
    lines = open(data_file, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    logger.debug(f'Size of dataset: {len(pairs)}, example data: {pairs[len(pairs)//2]}')
    return pairs
