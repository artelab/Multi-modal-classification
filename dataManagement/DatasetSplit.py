from tqdm import tqdm


class DatasetSplit(object):

    def __init__(self):
        self.texts = []
        self.labels = []
        self.images = []

    def load_data(self, data_file):
        with open(data_file) as tr:
            for line in tqdm(tr.readlines()):
                line = line.replace("\n", "")
                line = line.split('|')
                self.texts.append(line[0])
                self.labels.append(line[1])
                self.images.append(line[2])

    def get_texts(self):
        return self.texts

    def get_labels(self):
        return self.labels

    def get_images(self):
        return self.images
