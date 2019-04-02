from keras.preprocessing.text import Tokenizer


class TextTokenizer(object):

    def __init__(self, no_of_words_to_keep):
        self.tokenizer = Tokenizer(no_of_words_to_keep)

    def train_tokenizer(self, text):
        self.tokenizer.fit_on_texts(text)

    def convert_to_indices(self, text):
        return self.tokenizer.texts_to_sequences(text)
