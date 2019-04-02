import os
import pickle

from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from view.View import View


class BestDataHelperEverCreated:

    def __init__(self, num_words_to_keep, directory_of_data):
        self.view = View()
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse=False)
        self.tokenizer = Tokenizer(num_words_to_keep)
        self.directory_of_data = directory_of_data

    def train_tokenizer(self, text):
        self.tokenizer.fit_on_texts(text)

    def pickle_models_to_disk(self):
        with open(os.path.join(self.directory_of_data, 'label_encoder.pickle'), 'wb') as handle:
            pickle.dump(self.label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.directory_of_data, 'one_hot_encoder.pickle'), 'wb') as handle:
            pickle.dump(self.one_hot_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.directory_of_data, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_and_set_pickles(self):
        label_encoder_pathname = os.path.join(self.directory_of_data, 'label_encoder.pickle')
        self.label_encoder = self.load_pickle(label_encoder_pathname)

        one_hot_encoder_pathname = os.path.join(self.directory_of_data, 'one_hot_encoder.pickle')
        self.one_hot_encoder = self.load_pickle(one_hot_encoder_pathname)

        tokenizer_pathname = os.path.join(self.directory_of_data, 'tokenizer.pickle')
        self.tokenizer = self.load_pickle(tokenizer_pathname)

        return self.label_encoder, self.one_hot_encoder, self.tokenizer

    def convert_to_indices(self, text):
        return self.tokenizer.texts_to_sequences(text)

    def train_one_hot_encoder(self, train_labels):
        integer_encoded = self.label_encoder.fit_transform(train_labels)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        integer_encoded = integer_encoded.tolist()
        self.one_hot_encoder.fit_transform(integer_encoded)
        return self.one_hot_encoder, self.label_encoder

    def encode_to_one_hot(self, labels_list):
        integer_encoded = self.label_encoder.transform(labels_list)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        return self.one_hot_encoder.transform(integer_encoded)

    @staticmethod
    def load_pickle(pickle_pathname):
        with open(pickle_pathname, 'rb') as handle:
            return pickle.load(handle)
