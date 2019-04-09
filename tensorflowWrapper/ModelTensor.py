class ModelTensor(object):

    def __init__(self, input_x, input_y, input_mask, dropout_keep_prob):
        self.input_x = input_x
        self.input_y = input_y
        self.input_mask = input_mask
        self.dropout_keep_prob = dropout_keep_prob
