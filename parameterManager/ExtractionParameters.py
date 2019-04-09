import math


class ExtractionParameters(object):

    def __init__(self, output_image_width, encoding_height, ste_image_w, ste_separator_size, ste_superpixel_size):
        self.encoding_height = encoding_height
        self.output_image_width = output_image_width
        self.num_features = output_image_width * encoding_height
        self.ste_image_w = ste_image_w
        self.ste_image_h = ste_image_w
        self.separator_size = ste_separator_size
        self.superpixel_w = ste_superpixel_size
        self.superpixel_h = ste_superpixel_size
        self.superpixels_per_row = (self.ste_image_w - 2 * self.separator_size) / self.superpixel_w
        self.superpixels_per_col = math.ceil(self.num_features / self.superpixels_per_row)

    def get_encoding_height(self):
        return self.encoding_height

    def get_output_image_width(self):
        return self.output_image_width

    def get_num_features(self):
        return self.num_features

    def get_image_w(self):
        return self.ste_image_w

    def get_image_h(self):
        return self.ste_image_h

    def get_separator_size(self):
        return self.separator_size

    def get_superpixel_w(self):
        return self.superpixel_w

    def get_superpixel_h(self):
        return self.superpixel_h

    def get_superpixel_per_row(self):
        return self.superpixels_per_row

    def get_superpixel_per_col(self):
        return self.superpixels_per_col
