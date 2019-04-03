import os
import pickle

from labelManagement.DatasetLabelEncoder import DatasetLabelEncoder
from modelSaver.ModelPickler import ModelPickler
from textManagement.TextTokenizer import TextTokenizer
from view.View import View


class DataHelper:

    def __init__(self, num_words_to_keep, directory_of_data):
        self.view = View()
        self.label_encoder = DatasetLabelEncoder()
        self.tokenizer = TextTokenizer(num_words_to_keep)
        self.model_pickler = ModelPickler()
        self.directory_of_data = directory_of_data
        self.label_encoder_filename = os.path.join(self.directory_of_data, 'label_encoder.pickle')
        self.tokenizer_filename = os.path.join(self.directory_of_data, 'tokenizer.pickle')

    def train_one_hot_encoder(self, train_labels):
        return self.label_encoder.train_one_hot_encoder(train_labels)

    def train_tokenizer(self, train_texts):
        self.tokenizer.train_tokenizer(train_texts)

    def texts_to_indices(self, texts):
        return self.tokenizer.convert_to_indices(texts)

    def labels_to_one_hot(self, labels):
        return self.label_encoder.encode_to_one_hot(labels)

    def pickle_models_to_disk(self):
        self.model_pickler.pickle_models_to_disk(self.label_encoder, self.label_encoder_filename)
        self.model_pickler.pickle_models_to_disk(self.tokenizer, self.tokenizer_filename)

    def load_from_pickles(self):
        self.label_encoder = self.model_pickler.load_pickle(self.label_encoder_filename)
        self.tokenizer = self.model_pickler.load_pickle(self.tokenizer_filename)
