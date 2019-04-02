import os
import pickle


class ModelPickler(object):

    def __init__(self):
        pass

    @staticmethod
    def pickle_models_to_disk(object_to_save, object_filename):
        with open(object_filename, 'wb') as handle:
            pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle(pickle_pathname):
        with open(pickle_pathname, 'rb') as handle:
            return pickle.load(handle)
