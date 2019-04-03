import tensorflow as tf


class CustomIterator(object):

    def __init__(self):
        pass

    @staticmethod
    def create_iterator(dataset_split, batch_size):
        texts = dataset_split.get_texts()
        labels = dataset_split.get_labels()
        images = dataset_split.get_images()

        dataset = tf.data.Dataset.from_tensor_slices((texts, labels, images))
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        return iterator, next_element
