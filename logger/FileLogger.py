class FileLogger(object):

    def __init__(self, filename):
        self.filename = filename

    def write_header(self, model_directory):
        with open(self.filename, 'a') as resfile:
            resfile.write('Model dir: {}\n'.format(model_directory))

    def write_partial_result_to_file(self, partial_result):
        with open(self.filename, 'a') as resfile:
            resfile.write(str(partial_result))
