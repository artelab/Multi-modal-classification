import tensorflow as tf


class FlagsGenerator(object):

    def __init__(self):
        self.flags = None

    def get_flags(self):
        return self.flags

    def parse_parameter_file(self, filename):
        with open(filename) as parameter_file:
            for line in parameter_file.readlines():
                line = line.replace('\n', "")
                line = line.split(';')
                if line[2] == 'str':
                    tf.flags.DEFINE_string(line[0], line[1], '')
                elif line[2] == 'int':
                    tf.flags.DEFINE_integer(line[0], line[1], '')
                elif line[2] == 'float':
                    tf.flags.DEFINE_float(line[0], line[1], '')
            self.flags = tf.flags.FLAGS
