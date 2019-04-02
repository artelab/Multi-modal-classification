from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class DatasetLabelEncoder(object):

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.one_hot_encoder = OneHotEncoder(sparse=False)

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


