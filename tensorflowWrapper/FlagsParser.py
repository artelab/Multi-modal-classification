import tensorflow as tf


class FlagsParser(object):

    def __init__(self, delimiter):
        self.flags = None
        self.delimiter = delimiter

    def get_flags(self):
        return self.flags

    def parse_parameter_from_file(self, filename):
        with open(filename) as parameter_file:
            for line in parameter_file.readlines():
                tokens = self.sanitize_line(line)
                parameter_name = tokens[0]
                parameter_value = tokens[1]
                typology = tokens[2]

                if typology == 'str':
                    tf.flags.DEFINE_string(parameter_name, parameter_value, '')
                elif typology == 'int':
                    tf.flags.DEFINE_integer(parameter_name, parameter_value, '')
                elif typology == 'float':
                    tf.flags.DEFINE_float(parameter_name, parameter_value, '')
            self.flags = tf.flags.FLAGS

    def sanitize_line(self, line):
        line = line.replace('\n', '')
        return line.split(self.delimiter)
