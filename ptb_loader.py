import collections
import os

import numpy as np
from six.moves import cPickle


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
        self.go_token = '>'
        self.pad_token = '_'
        self.stop_token = '<'

        self.data_files = [data_path + path for path in ['ptb.test.txt', 'ptb.train.txt', 'ptb.valid.txt']]
        self.target_idx = {'test': 0, 'train': 1, 'valid': 2}

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
        idx_to_char = [self.go_token, self.stop_token, self.pad_token] + list(sorted(idx_to_char))

        char_to_idx = {x: i for i, x in enumerate(idx_to_char)}

        vocab_size = len(idx_to_char)

        return vocab_size, idx_to_char, char_to_idx

    def preprocess(self):

        self.data = [open(file, "r").read() for file in self.data_files]

        self.vocab_size, self.idx_to_char, self.char_to_idx = self.build_vocab(' '.join(self.data))

        self.data = [[line[1:-1] for line in target.split('\n')[:-1]] for target in self.data]

        self.num_lines = [len(target) for target in self.data]

        self.data = [[[self.char_to_idx[char]
                       for char in line]
                      for line in target]
                     for target in self.data]

        with open(self.idx_file, 'wb') as f:
            cPickle.dump(self.idx_to_char, f)

        with open(self.tensor_file, 'wb') as f:
            cPickle.dump(self.data, f)

    def load_preprocessed(self):

        self.idx_to_char = cPickle.load(open(self.idx_file, "rb"))
        self.vocab_size = len(self.idx_to_char)
        self.char_to_idx = dict(zip(self.idx_to_char, range(self.vocab_size)))

        self.data = cPickle.load(open(self.tensor_file, "rb"))

        self.num_lines = [len(target) for target in self.data]

    def next_batch(self, batch_size, target):
        """
        :param batch_size: number of selected data elements
        :param target: target from ['test', 'train', 'valid']
        :param use_cuda: whether to use cuda
        :return: target tensors
        """

        target = self.target_idx[target]

        indexes = np.random.randint(self.num_lines[target], size=batch_size)
        lines = np.array([self.data[target][idx][:] for idx in indexes])
        del indexes

        return self.construct_batches(lines)

    @staticmethod
    def sort_sequences(xs):
        """
        :param xs: An array of batches with length batch_size
        :return: Sorted array of batches
        """

        argsort = np.argsort([len(batch) for batch in xs])[::-1]
        return xs[argsort]

    def construct_batches(self, lines):
        """
        :param lines: An list of indexes arrays
        :return: Batches
        """

        lines = self.sort_sequences(lines)

        encoder_input = [self.add_token(line, go=True, stop=True) for line in lines]
        decoder_input = [self.add_token(line, go=True) for line in lines]
        decoder_target = [self.add_token(line, stop=True) for line in lines]

        encoder_input = self.padd_sequences(encoder_input)
        decoder_input = self.padd_sequences(decoder_input)
        decoder_target = self.padd_sequences(decoder_target)

        return encoder_input, decoder_input, decoder_target

    def padd_sequences(self, sequences):

        lengths = [len(line) for line in sequences]
        max_length = max(lengths)

        return np.array([line + [self.char_to_idx[self.pad_token]] * (max_length - lengths[i])
                         for i, line in enumerate(sequences)]), lengths

    def add_token(self, line, go=False, stop=False):

        go = [self.char_to_idx[self.go_token]] if go else []
        stop = [self.char_to_idx[self.stop_token]] if stop else []

        return go + line + stop
