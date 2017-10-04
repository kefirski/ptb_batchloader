import os
import re
import collections
from six.moves import cPickle



class PTBLoader():
    def __init__(self, data_path='', force_preprocessing=False):
        """
        :param data_path: path to PTB data
        :param force_preprocessing: wether to preprocess data even if it was preprocessed before
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
        self.pad_token = ''
        self.stop_token = '<'

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
        idx_to_char = [self.go_token, self.stop_token, self.pad_token] + list(sorted(idx_to_char))

        char_to_idx = {x: i for i, x in enumerate(idx_to_char)}

        vocab_size = len(idx_to_char)

        return vocab_size, idx_to_char, char_to_idx

    def preprocess(self):

        self.data = [open(file, "r").read() for file in self.data_files]

        self.vocab_size, self.idx_to_char, self.char_to_idx = self.build_vocab(' '.join(self.data))

        self.data = [[line[1:-1] for line in target.split('\n')[:-1]] for target in self.data]

        self.num_lines = [len(target) for target in self.data]

        with open(self.idx_file, 'wb') as f:
            cPickle.dump(self.idx_to_char, f)

        self.data = [[[self.char_to_idx[char]
                       for char in line]
                      for line in target]
                     for target in self.data]

        with open(self.tensor_file, 'wb') as f:
            cPickle.dump(self.data, f)

    def load_preprocessed(self):
        pass