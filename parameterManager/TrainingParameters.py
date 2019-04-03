class TrainingParameters(object):

    def __init__(self, no_of_words_to_keep, output_image_width, encoding_height, dropout, embedding_dim,
                 batch_size, filter_sizes, num_filters):
        self.no_of_words_to_keep = no_of_words_to_keep
        self.output_image_width = output_image_width
        self.encoding_height = encoding_height
        self.dropout_keep_probability = dropout
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

    def get_no_of_words_to_keep(self):
        return self.no_of_words_to_keep

    def get_output_image_width(self):
        return self.output_image_width

    def get_encoding_height(self):
        return self.encoding_height

    def get_dropout_keep_probability(self):
        return self.dropout_keep_probability

    def get_embedding_dim(self):
        return self.embedding_dim

    def get_batch_size(self):
        return self.batch_size

    def get_filter_sizes(self):
        return self.filter_sizes

    def get_num_filters(self):
        return self.num_filters
