import ast
import collections
import os

import numpy as np
import pandas as pnd
import torch as t
from six.moves import cPickle
from torch.autograd import Variable


class PTBLoader():
    def __init__(self, data_path='', force_preprocessing=False):
        """
        :param data_path: path to PTB data
        :param force_preprocessing: whether to preprocess data even if it was preprocessed before
        """

        assert isinstance(data_path, str), \
            'Invalid data_path type. Required {}, but {} found'.format(str, type(data_path))

        self.data_path = data_path
        self.preprocessings_path = self.data_path + 'preprocessings/'

        if not os.path.exists(self.preprocessings_path):
            os.makedirs(self.preprocessings_path)

        '''
        go_token (stop_token) uses to mark start (end) of the sequence
        pad_token uses to fill tensor to fixed-size length
        '''
        self.go_token = '~'
        self.pad_token = '_'
        self.stop_token = '|'

        self.data_files = [data_path + path for path in ['ptb.test.txt', 'ptb.train.txt', 'ptb.valid.txt']]

        self.idx_file = self.preprocessings_path + 'ptb_vocab.pkl'
        self.tensor_file = self.preprocessings_path + 'tensor.pkl'

        idx_exists = os.path.exists(self.idx_file)
        tensor_exists = os.path.exists(self.tensor_file)

        preprocessings_exist = all([file for file in [idx_exists, tensor_exists]])

        if preprocessings_exist and not force_preprocessing:

            print('Loading preprocessed data have started')
            self.load_preprocessed()
            print('Preprocessed data have loaded')
        else:

            print('Processing have started')
            self.preprocess()
            print('Data have preprocessed')

    def build_vocab(self, sentences):
        """
        :param sentences: An array of chars in data
        :return:
            vocab_size – Number of unique words in corpus
            idx_to_char – Array of shape [vocab_size] containing list of unique chars
            char_to_idx – Dictionary of shape [vocab_size]
                such that idx_to_char[char_to_idx[some_char]] = some_char
                where some_char is is from idx_to_char
        """

        char_counts = collections.Counter(sentences)

        idx_to_char = [x[0] for x in char_counts.most_common()]
        idx_to_char = [self.pad_token, self.go_token, self.stop_token] + list(sorted(idx_to_char))

        char_to_idx = {x: i for i, x in enumerate(idx_to_char)}

        vocab_size = len(idx_to_char)

        return vocab_size, idx_to_char, char_to_idx

    def preprocess(self):

        data = [open(file, "r").read() for file in self.data_files]

        self.vocab_size, self.idx_to_char, self.char_to_idx = self.build_vocab(' '.join(data))

        data = [[line[1:-1] for line in target.split('\n')[:-1]] for target in data]
        self.data = {
            target: pnd.DataFrame({
                'text': [self.go_token + line + self.stop_token for line in data[i]],
                'len': [len(line) for line in data[i]]
            })
            for i, target in enumerate(['test', 'train', 'valid'])
        }
        del data

        for target in self.data:
            self.data[target]['text'] = self.data[target]['text'].map(
                lambda line: [self.char_to_idx[char] for char in line]
            )

        with open(self.idx_file, 'wb') as f:
            cPickle.dump(self.idx_to_char, f)

        with open(self.tensor_file, 'wb') as f:
            cPickle.dump(self.data, f)

    def load_preprocessed(self):

        self.idx_to_char = cPickle.load(open(self.idx_file, "rb"))
        self.vocab_size = len(self.idx_to_char)
        self.char_to_idx = dict(zip(self.idx_to_char, range(self.vocab_size)))

        self.data = cPickle.load(open(self.tensor_file, "rb"))

        for target in self.data:
            self.data[target]['text'] = self.data[target]['text'].map(lambda line: ast.literal_eval(line))

    def next_batch(self, batch_size, target):
        """
        :param batch_size: number of selected data elements
        :param target: target from ['test', 'train', 'valid']
        :return: target tensors
        """

        indexes = np.random.choice(list(self.data[target].index), size=batch_size)
        lines = self.data[target].ix[indexes].sort_values('len', ascending=False)
        lines = list(lines['text'])
        del indexes

        return self.construct_batches(lines)

    def torch_batch(self, batch_size, target, cuda, volatile=False):

        (input, lengths), (gen_input, gen_lengths), (target, _) = self.next_batch(batch_size, target)
        [input, gen_input, target] = [Variable(t.from_numpy(var), volatile=volatile)
                                      for var in [input, gen_input, target]]
        if cuda:
            [input, gen_input, target] = [var.cuda() for var in [input, gen_input, target]]

        return (input, lengths), (gen_input, gen_lengths), target

    def construct_batches(self, lines):
        """
        :param lines: An list of indexes arrays
        :return: Batches
        """

        encoder_input = lines
        decoder_input = [line[:-1] for line in lines]
        decoder_target = [line[1:] for line in lines]

        encoder_input = self.padd_sequences(encoder_input)
        decoder_input = self.padd_sequences(decoder_input)
        decoder_target = self.padd_sequences(decoder_target)

        return encoder_input, decoder_input, decoder_target

    def padd_sequences(self, sequences):

        lengths = [len(line) for line in sequences]
        max_length = max(lengths)

        return np.array([line + [self.char_to_idx[self.pad_token]] * (max_length - lengths[i])
                         for i, line in enumerate(sequences)]), lengths

    def go_input(self, batch_size, use_cuda):

        go_input = np.array([[self.char_to_idx[self.go_token]]] * batch_size)
        go_input = Variable(t.from_numpy(go_input)).long()

        if use_cuda:
            go_input = go_input.cuda()

        return go_input

    def sample_char(self, p):
        """
        :param p: An array of probabilities
        :return: An index of sampled from distribution character
        """

        idx = np.random.choice(len(p), p=p.ravel())
        return idx, self.idx_to_char[idx]
