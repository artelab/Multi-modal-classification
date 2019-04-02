import pickle
import os

from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from tflearn.data_utils import pad_sequences
from tqdm import tqdm


class BestDataHelperEverCreated:

    def __init__(self, num_words_to_keep):
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse=False)
        self.tokenizer = Tokenizer(num_words_to_keep)

    def train_tokenizer(self, text):
        self.tokenizer.fit_on_texts(text)

    def pickle_everything_to_disk(self, dir):
        with open(os.path.join(dir, 'label_encoder.pickle'), 'wb') as handle:
            pickle.dump(self.label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(dir, 'one_hot_encoder.pickle'), 'wb') as handle:
            pickle.dump(self.one_hot_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(dir, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_all_pickles(dir):
        with open(os.path.join(dir, 'label_encoder.pickle'), 'rb') as handle:
            label_encoder = pickle.load(handle)
        with open(os.path.join(dir, 'one_hot_encoder.pickle'), 'rb') as handle:
            one_hot_encoder = pickle.load(handle)
        with open(os.path.join(dir, 'tokenizer.pickle'), 'rb') as handle:
            tokenizer = pickle.load(handle)
        return label_encoder, one_hot_encoder, tokenizer

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

    def prepare_for_tensorflow(self, text_list, labels_list, path_list, num_words_x_doc):
        text_list = [element.decode('UTF-8') for element in text_list]
        labels_list = [element.decode('UTF-8') for element in labels_list]
        path_list = [element.decode('UTF-8') for element in path_list]

        text_indices = self.tokenizer.texts_to_sequences(text_list)

        text_indices = pad_sequences(text_indices, maxlen=num_words_x_doc, value=0.)
        integer_encoded = self.label_encoder.transform(labels_list)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        y_one_hot = self.one_hot_encoder.transform(integer_encoded)

        return text_indices, y_one_hot, path_list

    def load_shuffle_data(self, train_path, val_path):
        print("Loading data...")

        text_train, label_train, img_train = self.load_data(train_path)
        text_val, label_val, img_val = self.load_data(val_path)

        text_train, label_train, img_train = shuffle(text_train, label_train, img_train, random_state=10)

        print("Train/Dev split: {:d}/{:d}".format(len(text_train), len(text_val)))
        return text_train, label_train, img_train, text_val, label_val, img_val

    @staticmethod
    def load_data(train_file):
        text = []
        label = []
        img = []
        with open(train_file) as tr:
            for line in tqdm(tr.readlines()):
                line = line.replace("\n", "")
                line = line.split('|')
                text.append(line[0])
                label.append(line[1])
                img.append(line[2])
        return text, label, img
