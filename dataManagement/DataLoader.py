from sklearn.utils import shuffle

from dataManagement.DatasetSplit import DatasetSplit
from view.View import View


class DataLoader(object):

    def __init__(self):
        self.view = View()
        self.train = DatasetSplit()
        self.val = DatasetSplit()

    def load_data(self, train_path, val_path, delimiter, shuffle_data=False):
        self.view.print_to_screen('Loading data...')

        self.train.load_data(train_path, delimiter)
        train_texts = self.train.get_texts()
        train_labels = self.train.get_labels()
        train_images = self.train.get_images()

        self.val.load_data(val_path, delimiter)
        val_texts = self.val.get_texts()
        val_labels = self.val.get_labels()
        val_images = self.val.get_images()

        self.view.print_to_screen("Train/Dev split: {:d}/{:d}".format(len(train_texts), len(val_texts)))

        if shuffle_data:
            train_texts, train_labels, train_images = shuffle(train_texts, train_labels, train_images, random_state=10)

        self.set_training_data(train_images, train_labels, train_texts)
        self.set_val_data(val_images, val_labels, val_texts)

    def set_val_data(self, val_images, val_labels, val_texts):
        self.val.set_texts(val_texts)
        self.val.set_labels(val_labels)
        self.val.set_images(val_images)

    def set_training_data(self,train_texts, train_labels, train_images):
        self.train.set_texts(train_texts)
        self.train.set_labels(train_labels)
        self.train.set_images(train_images)

    def get_training_data(self):
        return self.train

    def get_val_data(self):
        return self.val
