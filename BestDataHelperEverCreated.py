import os
import pickle

from labelManagement.DatasetLabelEncoder import DatasetLabelEncoder
from textManagement.TextTokenizer import TextTokenizer
from view.View import View


class BestDataHelperEverCreated:

    def __init__(self, num_words_to_keep, directory_of_data):
        self.view = View()
        self.label_encoder = DatasetLabelEncoder()
        self.tokenizer = TextTokenizer(num_words_to_keep)
        self.directory_of_data = directory_of_data
        self.label_encoder_filename = os.path.join(self.directory_of_data, 'label_encoder.pickle')
        self.one_hot_encoder_filename = os.path.join(self.directory_of_data, 'one_hot_encoder.pickle')
        self.tokenizer_filename = os.path.join(self.directory_of_data, 'tokenizer.pickle')

    def train_one_hot_encoder(self, train_labels):
        self.label_encoder.train_one_hot_encoder(train_labels)

    def train_tokenizer(self, train_texts):
        self.tokenizer.train_tokenizer(train_texts)

    def convert_to_indices(self, train_texts):
        return self.tokenizer.convert_to_indices(train_texts)

    def pickle_models_to_disk(self):
        with open(self.label_encoder_filename, 'wb') as handle:
            pickle.dump(self.label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.one_hot_encoder_filename, 'wb') as handle:
            pickle.dump(self.one_hot_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.tokenizer_filename, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_and_set_pickles(self):
        label_encoder_pathname = os.path.join(self.directory_of_data, 'label_encoder.pickle')
        self.label_encoder = self.load_pickle(label_encoder_pathname)

        one_hot_encoder_pathname = os.path.join(self.directory_of_data, 'one_hot_encoder.pickle')
        self.one_hot_encoder = self.load_pickle(one_hot_encoder_pathname)

        tokenizer_pathname = os.path.join(self.directory_of_data, 'tokenizer.pickle')
        self.tokenizer = self.load_pickle(tokenizer_pathname)

        return self.label_encoder, self.one_hot_encoder, self.tokenizer

    @staticmethod
    def load_pickle(pickle_pathname):
        with open(pickle_pathname, 'rb') as handle:
            return pickle.load(handle)

    def encode_to_one_hot(self, labels):
        return self.label_encoder.encode_to_one_hot(labels)
