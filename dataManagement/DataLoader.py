from sklearn.utils import shuffle

from dataManagement.DatasetSplit import DatasetSplit
from view.View import View


class DataLoader(object):

    def __init__(self):
        self.view = View()
        self.train = DatasetSplit()
        self.val = DatasetSplit()

    def load_shuffle_data(self, train_path, val_path):
        self.view.print_to_screen('Loading data...')

        self.train.load_data(train_path)
        train_texts = self.train.get_texts()
        train_labels = self.train.get_labels()
        train_images = self.train.get_images()
        text_train, label_train, img_train = shuffle(train_texts, train_labels, train_images, random_state=10)

        self.val.load_data(val_path)
        val_texts = self.val.get_texts()
        val_labels = self.val.get_labels()
        val_images = self.val.get_images()

        self.view.print_to_screen("Train/Dev split: {:d}/{:d}".format(len(text_train), len(val_texts)))
        return text_train, label_train, img_train, val_texts, val_labels, val_images

    def load_data(self, train_path):
        self.train
    
